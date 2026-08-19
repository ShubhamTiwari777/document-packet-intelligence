from __future__ import annotations
import unittest
from pathlib import Path

from src.config import AppConfig
from src.domain import Chunk, PageRepresentation, TextBlock
from src.features.extractor import FEATURE_NAMES, pair_features
from src.stage1.grouping import group_pages
from src.stage1.document_classifier import (
    HybridDocumentClassifier, LexiconDocumentClassifier, build_document_classifier,
)
from src.stage2.chunker import chunk_document
from src.stage3.retriever import HybridRetriever
from src.domain import StructuredDocument, Section


class CoreTests(unittest.TestCase):
    def test_pair_features_are_named_and_complete(self):
        left = PageRepresentation(1, "Invoice\nPage 1", [TextBlock("Invoice", [0, 0, 50, 20], [14])], [{"size": 14}], 600, 800)
        right = PageRepresentation(2, "Invoice\nPage 2", [TextBlock("Invoice", [0, 0, 50, 20], [14])], [{"size": 14}], 600, 800)
        self.assertEqual(set(pair_features(left, right)), set(FEATURE_NAMES))

    def test_grouping_preserves_original_pages(self):
        groups = group_pages("packet", [1, 2, 3, 4], [0.1, 0.9, 0.2], 0.5)
        self.assertEqual([group.pages for group in groups], [[1, 2], [3, 4]])

    def test_section_chunking_keeps_page_refs(self):
        document = StructuredDocument("doc", "invoice", [1], {}, [Section("s", "Invoice", 1, "one two three four five", [1], [])])
        config = AppConfig().chunking; config.max_tokens = 2; config.overlap_tokens = 1
        chunks = chunk_document(document, config)
        self.assertEqual(chunks[0].page_refs, [1]); self.assertGreater(len(chunks), 1)

    def test_hybrid_retrieval_returns_traceability(self):
        chunks = [Chunk("a", "invoice", "invoice", "s", [1], "Invoice number INV-42 total 100"), Chunk("b", "resume", "resume", "s", [2], "Experience and skills")]
        results = HybridRetriever(chunks, AppConfig().retrieval).retrieve("What is invoice number INV-42?")
        self.assertEqual(results[0].doc_id, "invoice"); self.assertEqual(results[0].page_ref, [1])



class DocumentClassifierTests(unittest.TestCase):
    INVOICE = "INVOICE Invoice Number INV-2026-004471 Bill To Acme Subtotal Total Payment terms"
    PASSPORT = "TRAVEL DOCUMENT Passport No P9876543 Surname SHARMA Nationality IND Issuing authority"

    def test_lexicon_abstains_instead_of_guessing_on_unrelated_text(self):
        prediction = HybridDocumentClassifier(None, min_confidence=0.35).predict(
            ["The quick brown fox jumps over the lazy dog."])[0]
        self.assertEqual(prediction.label, "unknown")
        self.assertEqual(prediction.confidence, 0.0)

    def test_single_generic_phrase_does_not_produce_a_confident_label(self):
        # Regression: the old scorer reported "invoice" at 0.2+ off one generic word.
        prediction = HybridDocumentClassifier(None, min_confidence=0.35).predict(["Total amount 500"])[0]
        self.assertEqual(prediction.label, "unknown")
        self.assertLess(prediction.confidence, 0.35)

    def test_confidence_is_comparable_and_bounded(self):
        predictions = LexiconDocumentClassifier().predict([self.INVOICE, self.PASSPORT])
        for prediction in predictions:
            self.assertGreaterEqual(prediction.confidence, 0.0)
            self.assertLessEqual(prediction.confidence, 1.0)
        self.assertEqual([p.label for p in predictions], ["invoice", "passport"])

    def test_extension_class_survives_a_trained_model_that_cannot_represent_it(self):
        class StubTrained:
            def predict(self, texts):
                from src.stage1.document_classifier import ClassPrediction
                return [ClassPrediction("letter", 0.9, "trained", []) for _ in texts]
        prediction = HybridDocumentClassifier(StubTrained(), min_confidence=0.35).predict([self.PASSPORT])[0]
        self.assertEqual(prediction.label, "passport")
        self.assertEqual(prediction.source, "lexicon_extension")

    def test_missing_model_path_is_reported_not_silently_ignored(self):
        _, status = build_document_classifier(None, 0.35, "tfidf_logistic_regression")
        self.assertFalse(status["trained_model_loaded"])
        self.assertIn("warning", status)


class Stage2StructureTests(unittest.TestCase):
    def _page(self, number, blocks, height=800.0):
        return PageRepresentation(number, "\n".join(b.text for b in blocks), blocks, [{"size": 10}], 600, height)

    def test_running_header_is_detected_and_excluded(self):
        from src.stage2.boilerplate import detect_boilerplate, normalize
        pages = [
            self._page(1, [TextBlock("ACME CORP", [0, 10, 200, 30], [10]), TextBlock("Page 1", [0, 770, 100, 790], [8]), TextBlock("Body one", [0, 300, 400, 320], [10])]),
            self._page(2, [TextBlock("ACME CORP", [0, 10, 200, 30], [10]), TextBlock("Page 2", [0, 770, 100, 790], [8]), TextBlock("Body two", [0, 300, 400, 320], [10])]),
        ]
        boilerplate = detect_boilerplate(pages)
        self.assertIn(normalize("ACME CORP"), boilerplate)
        # Digits are normalized so "Page 1"/"Page 2" collapse to a single repeated key.
        self.assertIn(normalize("Page 1"), boilerplate)
        self.assertNotIn(normalize("Body one"), boilerplate)

    def test_consecutive_single_item_blocks_become_one_list(self):
        from src.stage2.list_detector import match_item, split_list_items
        self.assertEqual(match_item("- First item"), "First item")
        self.assertEqual(match_item("2. Second item"), "Second item")
        self.assertIsNone(match_item("Just a sentence"))
        self.assertEqual(split_list_items("- a\n- b\n- c"), ["a", "b", "c"])

    def test_heading_levels_rank_by_font_size(self):
        from src.stage2.heading_detector import heading_levels
        self.assertEqual(heading_levels([18.0, 12.0, 12.0, 10.0]), {18.0: 1, 12.0: 2, 10.0: 3})

    def test_text_only_heading_detection_for_ocr_input(self):
        from src.stage2.heading_detector import is_text_heading, text_heading_level
        self.assertTrue(is_text_heading("ACCOUNT SUMMARY"))
        self.assertTrue(is_text_heading("2.1 Scope of Work"))
        self.assertEqual(text_heading_level("2.1 Scope of Work"), 2)
        self.assertFalse(is_text_heading("This is an ordinary prose sentence that ends properly."))

    def test_chunker_keeps_tables_intact_and_scopes_page_refs(self):
        from src.domain import Element as El
        table = El(type="table", text="| a | b |\n| --- | --- |\n| 1 | 2 |", page=2, bbox=[], rows=[["a", "b"], ["1", "2"]])
        section = Section("s1", "Summary", 1, "", [1, 2], [El("paragraph", "alpha beta", 1, []), table], breadcrumb=["Summary"])
        document = StructuredDocument("d1", "invoice", [1, 2], {}, [section])
        config = AppConfig().chunking
        chunks = chunk_document(document, config)
        table_chunks = [c for c in chunks if "table" in c.element_types]
        self.assertEqual(len(table_chunks), 1)
        self.assertIn("| 1 | 2 |", table_chunks[0].text)
        self.assertEqual(table_chunks[0].page_refs, [2])  # cites only the page it came from
        self.assertTrue(all(c.text.startswith("Summary") for c in chunks))  # breadcrumb context

    def test_invoice_total_is_not_matched_from_subtotal(self):
        from src.stage2.type_specific import extract_type_fields
        fields = extract_type_fields("invoice", "Subtotal: 84,500\nGST (18%): 15,210\nTotal: 99,710")
        self.assertEqual(fields["total"], "99,710")
        self.assertEqual(fields["subtotal"], "84,500")

    def test_passport_number_requires_identifier_shape(self):
        from src.stage2.type_specific import extract_type_fields
        self.assertNotIn("passport_number", extract_type_fields("passport", "Identity and Passport Information"))
        self.assertEqual(extract_type_fields("passport", "Passport No: P9876543")["passport_number"], "P9876543")


if __name__ == "__main__": unittest.main()
