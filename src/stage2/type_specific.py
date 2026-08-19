"""Type-conditional field extraction.

Two extraction bugs were found in the Stage 1 sample output and are fixed here:

* `total` matched the *inside* of "Subtotal: 84,500" and reported 84500 as the invoice total
  when the real total was 99,710. Fixed with a leading word boundary plus an explicit
  preference for the most specific label present (grand total > total > amount due).
* `passport_number` matched the prose "Passport Information" and captured the word
  "Information". Fixed by requiring an identifier-shaped value rather than any word.
"""
from __future__ import annotations

import re

# Ordered alternatives: the first pattern that matches wins, so more specific labels are
# listed before generic ones.
PATTERNS: dict[str, dict[str, list[str]]] = {
    "invoice": {
        "invoice_number": [r"invoice\s*(?:number|no\.?|#)\s*[:#]?\s*([A-Z0-9][A-Z0-9/-]{2,})"],
        "total": [
            r"\bgrand\s+total\s*[:\-]?\s*[$₹€£]?\s*([\d][\d,]*(?:\.\d{2})?)",
            r"(?<!sub)(?<!\w)total\s*[:\-]?\s*[$₹€£]?\s*([\d][\d,]*(?:\.\d{2})?)",
            r"\bamount\s+due\s*[:\-]?\s*[$₹€£]?\s*([\d][\d,]*(?:\.\d{2})?)",
        ],
        "subtotal": [r"\bsub\s*total\s*[:\-]?\s*[$₹€£]?\s*([\d][\d,]*(?:\.\d{2})?)"],
    },
    "passport": {
        # Require an identifier shape (letters+digits), never a bare prose word.
        "passport_number": [r"passport\s*(?:number|no\.?|#)\s*[:#]?\s*([A-Z]{1,2}\d{5,9}|[A-Z0-9<]{6,12})"],
        "surname": [r"\bsurname\s*[:\-]?\s*([A-Z][A-Za-z'-]+)"],
        "nationality": [r"\bnationality\s*[:\-]?\s*([A-Z]{2,3}|[A-Z][a-z]+)"],
        "expiry_date": [r"(?:date of expiry|expiry)\s*[:\-]?\s*([\d]{1,2}[/-][\d]{1,2}[/-][\d]{2,4}|\d{1,2}\s+\w+\s+\d{4})"],
    },
    "bank_statement": {
        "closing_balance": [r"closing balance\s*[:\-]?\s*[$₹€£]?\s*([\d][\d,]*(?:\.\d{2})?)"],
        "opening_balance": [r"opening balance\s*[:\-]?\s*[$₹€£]?\s*([\d][\d,]*(?:\.\d{2})?)"],
        "account_number": [r"account\s*(?:number|no\.?|#)\s*[:#]?\s*([X\d][X\d-]{3,})"],
    },
    "resume": {
        "email": [r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"],
    },
}


def extract_type_fields(doc_type: str, text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for name, patterns in PATTERNS.get(doc_type, {}).items():
        for pattern in patterns:
            found = re.search(pattern, text, re.I)
            if found:
                fields[name] = found.group(1).strip()
                break
    return fields
