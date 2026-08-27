"""Generate a dense 13-page, 7-document stress packet for exercising the pipeline and the UI.

The shipped `sample_packet.pdf` is scored perfectly by the current models, which makes it useless
for telling whether a change helped or hurt. This packet is arranged to attack the specific
weaknesses the evaluation surfaced:

* An invoice and a budget sit adjacent, share a letterhead, and both lead with a ruled money
  table -- `budget` is the trained classifier's top confusion for `invoice`, so this tests the
  boundary detector where header and layout similarity actively mislead it.
* A one-page letter and a one-page memo sit back to back, so two consecutive seams are both real
  boundaries. Short documents are where page grouping measured below the trivial baseline.
* The passport runs three pages whose layouts differ sharply from each other (fields, then a
  ruled table, then prose), inviting a false split inside a single document.
* All seven types are nameable by the system -- four from the trained RVL-CDIP taxonomy
  (invoice, budget, letter, memo), one shared (resume), two lexicon extensions
  (bank_statement, passport) -- so classification accuracy is meaningful rather than
  capped by the taxonomy.

Emits the PDF plus a ground-truth JSON that can be pasted straight into the UI.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

try:  # PyMuPDF renamed its import; support both.
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

LETTERHEAD, HEADING, BODY, FOOT = 17, 11.5, 9.5, 7.5
LEFT, RIGHT = 64.0, 548.0


def _text(page, x, y, value, size=BODY, bold=False):
    page.insert_text((x, y), value, fontsize=size, fontname="hebo" if bold else "helv")


def _head(page, title, subtitle, label):
    _text(page, LEFT, 58, title, LETTERHEAD, bold=True)
    _text(page, LEFT, 75, subtitle, BODY)
    page.draw_line(fitz.Point(LEFT, 84), fitz.Point(RIGHT, 84))
    _text(page, RIGHT - 60, 770, label, FOOT)


def _section(page, y, title):
    _text(page, LEFT, y, title, HEADING, bold=True)
    return y + 18


def _para(page, y, value, width=104, size=BODY, lead=13):
    for line in textwrap.wrap(value, width):
        _text(page, LEFT, y, line, size)
        y += lead
    return y + 4


def _items(page, y, values, bullet="-", lead=14):
    for value in values:
        _text(page, LEFT + 6, y, f"{bullet} {value}" if bullet else value, BODY)
        y += lead
    return y + 4


def _table(page, top, rows, widths, row_height=17.0):
    xs = [LEFT]
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
            _text(page, xs[c] + 4, top + row_height * r + 12, str(cell), size=8.8, bold=(r == 0))
    return bottom + 10


def build_packet(output_path: str | Path) -> tuple[Path, dict]:
    doc = fitz.open()
    truth: list[dict] = []
    new = lambda: doc.new_page(width=612, height=792)

    VENDOR = "Northwind Logistics Pvt. Ltd."

    # ================= 1. INVOICE (pages 1-2) =================
    page = new(); _head(page, "TAX INVOICE", VENDOR, "Page 1 of 2")
    y = _section(page, 112, "Billing Details")
    y = _items(page, y, [
        "Invoice Number: NWL-INV-2026-08841", "Invoice Date: 12 August 2026",
        "Purchase Order: PO-4471-B", "Bill To: Harbourline Retail Group, Mumbai",
        "Payment Due: 11 September 2026",
    ], bullet="")
    y = _section(page, y + 8, "Line Items")
    y = _table(page, y, [
        ["Description", "Qty", "Unit Price", "Amount"],
        ["Freight forwarding - Nhava Sheva to Chennai", "12", "18,400", "2,20,800"],
        ["Customs clearance and documentation", "12", "4,250", "51,000"],
        ["Warehousing (30 days, 400 sq ft)", "1", "68,000", "68,000"],
        ["Last-mile distribution - Tamil Nadu", "9", "7,150", "64,350"],
        ["Cargo insurance premium", "1", "22,300", "22,300"],
    ], widths=[268, 42, 88, 88])
    y = _section(page, y + 6, "Payment Terms")
    _items(page, y, [
        "Net 30 days from invoice date.",
        "Late payments attract interest at 1.5% per month.",
        "Remit to Northwind Logistics, A/C 91882140055, IFSC NWLB0004412.",
    ])

    page = new(); _head(page, "TAX INVOICE", VENDOR, "Page 2 of 2")
    y = _section(page, 112, "Summary of Charges")
    y = _table(page, y, [
        ["Component", "Amount"],
        ["Subtotal", "4,26,450"],
        ["GST (18%)", "76,761"],
        ["Round off", "-11"],
        ["Total Amount Due", "5,03,200"],
    ], widths=[330, 156])
    y = _section(page, y + 6, "Notes and Terms")
    y = _items(page, y, [
        "Services rendered per the master service agreement dated 04 April 2026.",
        "Disputes must be raised in writing within 14 days of the invoice date.",
        "Goods remain the property of the consignor until payment clears in full.",
        "This is a computer-generated invoice and needs no signature.",
    ], bullet=None)
    _para(page, y + 6, "For queries regarding this invoice contact accounts receivable at "
          "receivables@northwindlogistics.example or call the billing desk on 022-4471-0088 "
          "between 10:00 and 18:00 IST on working days.")
    truth.append({"pages": [1, 2], "doc_type": "invoice", "title": "TAX INVOICE"})

    # ================= 2. BUDGET (pages 3-4) — same letterhead, also money tables =========
    page = new(); _head(page, "DEPARTMENTAL BUDGET", VENDOR, "Page 1 of 2")
    y = _section(page, 112, "Budget Allocation FY 2026-27")
    y = _para(page, y, "The following allocation was approved by the finance committee for the "
              "logistics operations department. Figures are in Indian rupees and exclude "
              "capital expenditure carried forward from the previous financial year.")
    y = _table(page, y, [
        ["Cost Centre", "Approved", "Committed", "Available"],
        ["Fleet operations", "1,84,00,000", "1,12,40,000", "71,60,000"],
        ["Warehouse leasing", "96,50,000", "96,50,000", "0"],
        ["Personnel and training", "1,42,75,000", "88,10,000", "54,65,000"],
        ["Technology and systems", "58,20,000", "21,90,000", "36,30,000"],
        ["Contingency reserve", "35,00,000", "0", "35,00,000"],
    ], widths=[186, 100, 100, 100])
    y = _section(page, y + 6, "Planning Assumptions")
    _items(page, y, [
        "Diesel priced at an average of 94.50 per litre across the year.",
        "Headcount grows from 214 to 240 by the fourth quarter.",
        "No new warehouse leases are signed before Q3.",
        "Contingency is released only on finance-committee approval.",
    ])

    page = new(); _head(page, "DEPARTMENTAL BUDGET", VENDOR, "Page 2 of 2")
    y = _section(page, 112, "Quarterly Phasing")
    y = _table(page, y, [
        ["Cost Centre", "Q1", "Q2", "Q3", "Q4"],
        ["Fleet operations", "46,00,000", "46,00,000", "46,00,000", "46,00,000"],
        ["Warehouse leasing", "24,12,500", "24,12,500", "24,12,500", "24,12,500"],
        ["Personnel and training", "31,00,000", "34,25,000", "37,50,000", "40,00,000"],
        ["Technology and systems", "9,20,000", "14,00,000", "18,00,000", "17,00,000"],
    ], widths=[150, 84, 84, 84, 84])
    y = _section(page, y + 6, "Variance Commentary")
    y = _para(page, y, "Warehouse leasing is fully committed at the start of the year because the "
              "Chennai and Pune facilities renew on annual terms. Technology spend is deliberately "
              "back-loaded pending the outcome of the transport-management-system evaluation.")
    y = _section(page, y + 4, "Approval")
    _items(page, y, [
        "Prepared by: R. Venkataraman, Finance Business Partner",
        "Reviewed by: S. Dhillon, Head of Logistics Operations",
        "Approved on: 28 March 2026",
    ], bullet="")
    truth.append({"pages": [3, 4], "doc_type": "budget", "title": "DEPARTMENTAL BUDGET"})

    # ================= 3. LETTER (page 5) =================
    page = new(); _head(page, "MERIDIAN LEGAL ASSOCIATES", "Advocates and Solicitors, Bengaluru", "")
    y = 112
    _text(page, RIGHT - 130, y, "19 August 2026", BODY); y += 26
    y = _items(page, y, ["The Company Secretary", "Harbourline Retail Group",
                         "Plot 14, Andheri East", "Mumbai 400069"], bullet="")
    y += 10
    _text(page, LEFT, y, "Dear Sir or Madam,", BODY); y += 22
    _text(page, LEFT, y, "Re: Renewal of the logistics services agreement", BODY, bold=True); y += 22
    y = _para(page, y, "We act for Northwind Logistics Pvt. Ltd. in connection with the master "
              "service agreement dated 4 April 2026. Our client has instructed us to write to you "
              "regarding the renewal provisions in clause 11 of that agreement.")
    y = _para(page, y, "Clause 11.2 requires either party to give not less than ninety days written "
              "notice of an intention not to renew. No such notice has been received by our client, "
              "and accordingly the agreement will renew automatically on 4 April 2027 on the terms "
              "presently in force, subject only to the indexation mechanism in schedule 3.")
    y = _para(page, y, "We would be grateful if you would confirm your agreement to this position in "
              "writing within twenty-one days. Should you consider that any part of this letter is "
              "incorrect, please set out your position with reference to the relevant clauses so "
              "that the matter can be resolved without recourse to the dispute procedure.")
    y += 10
    _text(page, LEFT, y, "Yours faithfully,", BODY); y += 34
    _text(page, LEFT, y, "A. Krishnamurthy", BODY, bold=True); y += 13
    _text(page, LEFT, y, "Partner, Meridian Legal Associates", BODY)
    truth.append({"pages": [5], "doc_type": "letter", "title": "MERIDIAN LEGAL ASSOCIATES"})

    # ================= 4. MEMO (page 6) — second single-page doc in a row =========
    page = new(); _head(page, "INTERNAL MEMORANDUM", "Harbourline Retail Group", "Confidential")
    y = _section(page, 112, "Memorandum")
    y = _items(page, y, [
        "TO: Regional Distribution Managers",
        "FROM: K. Iyer, Director of Supply Chain",
        "DATE: 22 August 2026",
        "SUBJECT: Interim changes to inbound freight routing",
    ], bullet="")
    y = _para(page, y + 6, "Effective 1 September, all inbound freight for the southern region is to "
              "be routed through the Chennai consolidation hub rather than direct-to-store. This is "
              "an interim measure while the Nhava Sheva berth allocation is under review.")
    y = _section(page, y + 4, "Action Required")
    y = _items(page, y, [
        "Update routing rules in the distribution system before 29 August.",
        "Notify carriers of the revised delivery windows.",
        "Report exceptions to the supply chain desk within 24 hours.",
        "Confirm completion by replying to this memorandum.",
    ], bullet=None)
    _para(page, y + 4, "This memorandum supersedes the routing guidance issued on 3 July 2026. "
          "Questions should be directed to the supply chain desk in the first instance.")
    truth.append({"pages": [6], "doc_type": "memo", "title": "INTERNAL MEMORANDUM"})

    # ================= 5. RESUME (pages 7-8) =================
    page = new(); _head(page, "CURRICULUM VITAE", "Priya Nair - Senior Data Engineer", "Page 1 of 2")
    y = _section(page, 112, "Professional Summary")
    y = _para(page, y, "Senior data engineer with nine years of experience building batch and "
              "streaming pipelines for retail and logistics organisations. Responsible for a "
              "platform serving 40 million events per day across twelve downstream teams.")
    y = _section(page, y + 4, "Technical Skills")
    y = _items(page, y, [
        "Languages: Python, Scala, SQL, Go",
        "Platforms: Spark, Kafka, Airflow, dbt, Snowflake",
        "Cloud: AWS (EMR, Glue, Redshift), Terraform, Kubernetes",
        "Practices: data modelling, lineage, cost optimisation, on-call leadership",
    ])
    y = _section(page, y + 4, "Work Experience")
    y = _para(page, y, "Harbourline Retail Group, Mumbai - Senior Data Engineer, 2021 to present. "
              "Led the migration from nightly batch to near-real-time ingestion, cutting reporting "
              "latency from 14 hours to under 20 minutes and reducing warehouse spend by 31%.")
    _para(page, y, "Cobalt Analytics, Pune - Data Engineer, 2017 to 2021. Built the customer data "
          "platform underpinning segmentation for a retail client with 8 million loyalty members.")

    page = new(); _head(page, "CURRICULUM VITAE", "Priya Nair - Senior Data Engineer", "Page 2 of 2")
    y = _section(page, 112, "Education")
    y = _table(page, y, [
        ["Qualification", "Institution", "Year"],
        ["M.Tech, Computer Science", "College of Engineering, Pune", "2017"],
        ["B.E., Information Technology", "University of Mumbai", "2015"],
    ], widths=[200, 200, 86])
    y = _section(page, y + 6, "Certifications")
    y = _items(page, y, [
        "AWS Certified Data Analytics - Specialty (2024)",
        "Databricks Certified Data Engineer Professional (2023)",
    ])
    y = _section(page, y + 4, "Selected Projects")
    y = _items(page, y, [
        "Real-time stock visibility across 340 stores, built on Kafka and Flink.",
        "Cost attribution model allocating warehouse spend to business units.",
        "Data quality framework with 600 automated expectations in production.",
    ])
    _para(page, y + 4, "References are available on request. Contact: priya.nair@example.com")
    truth.append({"pages": [7, 8], "doc_type": "resume", "title": "CURRICULUM VITAE"})

    # ================= 6. BANK STATEMENT (pages 9-10) =================
    page = new(); _head(page, "ACCOUNT STATEMENT", "Sample Bank - Corporate Banking", "Page 1 of 2")
    y = _section(page, 112, "Account Summary")
    y = _items(page, y, [
        "Account Holder: Northwind Logistics Pvt. Ltd.",
        "Account Number: XXXXXX91882140",
        "Account Type: Current Account",
        "Statement Period: 01 August 2026 to 31 August 2026",
        "Opening Balance: 82,41,300",
        "IFSC: NWLB0004412",
    ], bullet="")
    y = _section(page, y + 8, "Statement Notes")
    _items(page, y, [
        "All amounts are in Indian rupees unless stated otherwise.",
        "Cheques are credited subject to realisation.",
        "Report discrepancies within 30 days of the statement date.",
    ])

    page = new(); _head(page, "ACCOUNT STATEMENT", "Sample Bank - Corporate Banking", "Page 2 of 2")
    y = _section(page, 112, "Transaction Detail")
    y = _table(page, y, [
        ["Date", "Particulars", "Debit", "Credit", "Balance"],
        ["02 Aug", "NEFT - Harbourline Retail Group", "", "5,03,200", "87,44,500"],
        ["05 Aug", "Fuel card settlement", "3,18,600", "", "84,25,900"],
        ["09 Aug", "Payroll - August cycle", "24,80,000", "", "59,45,900"],
        ["14 Aug", "RTGS - Chennai warehouse lease", "8,04,167", "", "51,41,733"],
        ["21 Aug", "NEFT - Coastal Freight Ltd", "", "12,60,000", "64,01,733"],
        ["27 Aug", "GST remittance", "76,761", "", "63,24,972"],
    ], widths=[54, 216, 76, 76, 82])
    y = _section(page, y + 6, "Period Totals")
    _items(page, y, [
        "Total Debits: 36,79,528", "Total Credits: 17,63,200",
        "Closing Balance: 63,24,972",
    ], bullet="")
    truth.append({"pages": [9, 10], "doc_type": "bank_statement", "title": "ACCOUNT STATEMENT"})

    # ========= 7. PASSPORT (pages 11-13) — three sharply different layouts in one document =====
    page = new(); _head(page, "TRAVEL DOCUMENT", "Republic of India - Passport", "Page 1 of 3")
    y = _section(page, 112, "Holder Information")
    y = _items(page, y, [
        "Passport No: R4482913", "Surname: NAIR", "Given Names: PRIYA",
        "Nationality: IND", "Sex: F", "Date of Birth: 08 November 1993",
        "Place of Birth: KOCHI, KERALA", "Date of Issue: 17 January 2023",
        "Date of Expiry: 16 January 2033", "Place of Issue: MUMBAI",
    ], bullet="")
    y = _section(page, y + 8, "Issuing Authority")
    _para(page, y, "Issuing authority: Regional Passport Office, Mumbai. This travel document "
          "remains the property of the Government of India and must be surrendered on demand.")

    page = new(); _head(page, "TRAVEL DOCUMENT", "Republic of India - Passport", "Page 2 of 3")
    y = _section(page, 112, "Visa Records")
    y = _table(page, y, [
        ["Country", "Visa Type", "Issued", "Expires", "Entries"],
        ["Singapore", "Business", "04 Feb 2024", "03 Feb 2026", "Multiple"],
        ["United Kingdom", "Standard Visitor", "19 Jun 2024", "18 Jun 2026", "Multiple"],
        ["United Arab Emirates", "Visit", "02 Dec 2025", "01 Mar 2026", "Single"],
    ], widths=[136, 120, 84, 84, 62])
    y = _section(page, y + 6, "Immigration Endorsements")
    _table(page, y, [
        ["Port", "Direction", "Date"],
        ["Chhatrapati Shivaji, Mumbai", "Departure", "07 Feb 2024"],
        ["Changi, Singapore", "Arrival", "07 Feb 2024"],
        ["Heathrow, London", "Arrival", "24 Jun 2024"],
    ], widths=[240, 130, 116])

    page = new(); _head(page, "TRAVEL DOCUMENT", "Republic of India - Passport", "Page 3 of 3")
    y = _section(page, 112, "Observations")
    y = _para(page, y, "The holder is the spouse of ARJUN MENON, passport number M8871204. This "
              "endorsement is recorded under the provisions applicable to family particulars and "
              "does not itself confer any right of entry or residence.")
    y = _section(page, y + 4, "Emergency Contact")
    y = _items(page, y, [
        "Name: Lakshmi Nair", "Relationship: Mother",
        "Address: 22 Marine Drive, Kochi, Kerala 682031", "Telephone: +91 484 220 8814",
    ], bullet="")
    y = _section(page, y + 6, "Conditions of Use")
    _items(page, y, [
        "This document contains 36 pages including the cover.",
        "Alteration or mutilation of this document renders it invalid.",
        "Loss or theft must be reported to the nearest passport office immediately.",
        "This document is not valid for travel to countries under advisory restriction.",
    ], bullet=None)
    truth.append({"pages": [11, 12, 13], "doc_type": "passport", "title": "TRAVEL DOCUMENT"})

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    page_count = doc.page_count
    doc.close()

    ground_truth = {"packet": path.name, "page_count": page_count, "documents": truth}
    truth_path = path.with_name(f"{path.stem}_ground_truth.json")
    truth_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    return path, ground_truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/samples/stress_packet.pdf")
    args = parser.parse_args()
    path, truth = build_packet(args.output)
    boundaries = len(truth["documents"]) - 1
    seams = truth["page_count"] - 1
    print(f"Wrote {path}: {truth['page_count']} pages, {len(truth['documents'])} documents, "
          f"{boundaries}/{seams} seams are boundaries ({boundaries / seams:.0%} density)")
    for document in truth["documents"]:
        print(f"  pages {str(document['pages']):<14} {document['doc_type']}")
    print(f"\nGround truth for the UI -> {path.with_name(path.stem + '_ground_truth.json')}")
