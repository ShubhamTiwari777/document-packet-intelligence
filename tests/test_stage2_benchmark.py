"""Stage 2 regression tests against the generated benchmark report PDF.

These pin the specific defects found when the structured output was inspected through the API:
tables flattened into paragraph text, breadcrumbs frozen on an early heading, numbered headings
never detected, and captions/list ordering lost. Each test maps to one of those failures so they
cannot silently return.

The final test drives the real FastAPI endpoint, because every earlier bug was visible only in
the assembled API response -- unit-testing the parser modules alone did not surface them.
"""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.domain import DocumentGroup
from src.ingestion.pdf_parser import PDFParser
from src.stage2.chunker import chunk_document
from src.stage2.structure_parser import structure_document
from scripts.generate_benchmark_report import build


class Stage2BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = TemporaryDirectory()
        cls.pdf_path, cls.truth = build(Path(cls._temporary.name) / "benchmark_report.pdf")
        cls.config = load_config("config/default.yaml")
        cls.pages = PDFParser(cls.config.ingestion, cls.config.runtime).parse(str(cls.pdf_path), None)
        group = DocumentGroup(doc_id="bench_001", pages=[page.page_number for page in cls.pages], doc_type="report")
        cls.document = structure_document(group, cls.pages, cls.pdf_path.name, str(cls.pdf_path))
        cls.chunks = chunk_document(cls.document, cls.config.chunking)
        cls.elements = [element for section in cls.document.sections for element in section.elements]

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def _of_type(self, kind):
        return [element for element in self.elements if element.type == kind]

    # 1 + 2
    def test_at_least_two_tables_with_rows_and_columns(self):
        tables = self._of_type("table")
        self.assertGreaterEqual(len(tables), 2, "benchmark contains a ruled and a borderless table")
        for table in tables:
            self.assertTrue(table.rows, "table must retain rows")
            self.assertGreaterEqual(len(table.rows), 2)
            self.assertGreaterEqual(len(table.rows[0]), 2, "table must retain columns")
            self.assertEqual(table.headers, table.rows[0])
            self.assertIn("|", table.text, "table must render as markdown")

    # 3
    def test_tables_are_not_duplicated_as_paragraphs(self):
        table_cells = {cell for table in self._of_type("table") for row in table.rows for cell in row if cell.strip()}
        distinctive = [cell for cell in table_cells if len(cell) > 6]
        self.assertTrue(distinctive, "expected some distinctive table cell text")
        for paragraph in self._of_type("paragraph"):
            for cell in distinctive:
                self.assertNotIn(cell, paragraph.text, f"table cell {cell!r} leaked into a paragraph")

    # 4
    def test_breadcrumbs_track_the_current_section(self):
        breadcrumbs = [tuple(section.breadcrumb) for section in self.document.sections if section.breadcrumb]
        self.assertGreater(len(set(breadcrumbs)), 5, "breadcrumbs must change as sections change")
        # The specific reported symptom: later content stuck under the first subsection.
        titles = [section.title for section in self.document.sections]
        self.assertIn("3.3 Example Measurement Table", titles)
        target = next(s for s in self.document.sections if s.title == "3.3 Example Measurement Table")
        self.assertNotIn("Key objectives", target.breadcrumb)
        self.assertEqual(target.breadcrumb[-1], "3.3 Example Measurement Table")

    # 5
    def test_every_element_has_page_provenance(self):
        valid_pages = {page.page_number for page in self.pages}
        for element in self.elements:
            self.assertIn(element.page, valid_pages)
        for section in self.document.sections:
            # A section's page refs must be exactly the pages its elements came from.
            self.assertEqual(section.page_refs, sorted({element.page for element in section.elements}))
        for chunk in self.chunks:
            self.assertTrue(chunk.page_refs)

    # 6
    def test_lists_preserve_type_and_ordering(self):
        lists = self._of_type("list")
        self.assertGreaterEqual(len(lists), 3)
        self.assertTrue(any(item.ordered for item in lists), "numbered list must be marked ordered")
        self.assertTrue(any(item.ordered is False for item in lists), "bullet list must be marked unordered")
        ordered_list = next(item for item in lists if item.ordered)
        self.assertEqual(ordered_list.items[0], "Parse pages and blocks.", "item order must be preserved")
        for item in lists:
            self.assertGreaterEqual(len(item.items), 2)

    # 7
    def test_headings_produce_multiple_hierarchy_levels(self):
        levels = {section.level for section in self.document.sections}
        self.assertGreater(len(levels), 1, "numbered subsections must nest below their parent")
        numbered = next(s for s in self.document.sections if s.title.startswith("1.1"))
        self.assertEqual(numbered.level, 2)
        self.assertIsNotNone(numbered.parent_section_id)
        parents = {s.section_id for s in self.document.sections}
        # A referenced parent must be a real, distinct section id.
        for section in self.document.sections:
            if section.parent_section_id:
                self.assertNotEqual(section.parent_section_id, section.section_id)

    # 8
    def test_figure_caption_is_detected(self):
        captions = self._of_type("caption")
        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0].label, "Figure 1")
        self.assertIn("Conceptual flow", captions[0].text)

    # 9
    def test_repeated_headers_do_not_become_headings(self):
        titles = [section.title for section in self.document.sections]
        self.assertNotIn("Document Intelligence Evaluation Report", titles[1:], "running header became a heading")
        self.assertFalse([t for t in titles if t.startswith("Page ")], "page-number footer became a heading")
        self.assertTrue(self.document.metadata["boilerplate_lines"], "boilerplate should be recorded")
        for element in self.elements:
            self.assertNotEqual(element.text.strip(), "Document Intelligence Evaluation Report")

    # 10
    def test_fastapi_response_matches_the_parser_output(self):
        from fastapi.testclient import TestClient
        from src.api import create_app

        client = TestClient(create_app("config/default.yaml"))
        with open(self.pdf_path, "rb") as handle:
            response = client.post("/process", files={"file": (self.pdf_path.name, handle, "application/pdf")})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        for key in ("packet_id", "documents", "chunk_count", "output_dir", "model_status", "structured_documents", "chunks"):
            self.assertIn(key, payload, f"API response missing {key}")

        api_elements = [e for d in payload["structured_documents"] for s in d["sections"] for e in s["elements"]]
        self.assertEqual(Counter(e["type"] for e in api_elements), Counter(e.type for e in self.elements))
        api_tables = [e for e in api_elements if e["type"] == "table"]
        self.assertGreaterEqual(len(api_tables), 2)
        for table in api_tables:
            self.assertTrue(table["rows"] and table["headers"])
        self.assertTrue(all(chunk["page_refs"] for chunk in payload["chunks"]))
        self.assertTrue(all(chunk["token_count"] > 0 for chunk in payload["chunks"]))
        self.assertGreater(len({tuple(c["breadcrumb"]) for c in payload["chunks"]}), 5)


if __name__ == "__main__":
    unittest.main()
