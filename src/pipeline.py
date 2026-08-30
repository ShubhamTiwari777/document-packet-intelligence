"""Business orchestration used by CLI and API; no UI-specific logic lives here."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json

from src.config import AppConfig
from src.domain import Chunk, dump_json
from src.features.extractor import packet_feature_rows, save_feature_cache
from src.features.document_shape import document_shape_features, shape_vector
from src.features.text_features import TfidfTextEmbedder
from src.ingestion.pdf_parser import PDFParser
from src.stage1.boundary_classifier import WeightedFusionBoundary, SklearnBoundaryModel
from src.stage1.document_classifier import build_document_classifier
from src.stage1.grouping import group_pages
from src.stage2.chunker import chunk_document
from src.stage2.markdown_renderer import render_markdown
from src.stage2.structure_parser import structure_document
from src.stage3.context import assemble_context
from src.stage3.retriever import HybridRetriever


class DocumentPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self._retriever_cache: dict[tuple[str, int], HybridRetriever] = {}

    def process(self, input_pdf: str | Path, output_dir: str | Path, packet_id: str | None = None, boundary_model_dir: str | Path | None = None, document_model: str | Path | None = None, include_structure: bool = True, include_chunks: bool = True) -> dict[str, Any]:
        source = Path(input_pdf); target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
        packet = packet_id or source.stem
        pages = PDFParser(self.config.ingestion, self.config.runtime).parse(source, target / "rendered_pages")
        selected_boundary_model = boundary_model_dir or self.config.boundary.model_path
        text_embedder_path = Path(selected_boundary_model) / "text_embedder.pkl" if selected_boundary_model else None
        text_embedder = TfidfTextEmbedder.load(text_embedder_path) if text_embedder_path and text_embedder_path.exists() else None
        features = packet_feature_rows(pages, packet, text_embedder); save_feature_cache(features, target / "boundary_features.json")
        model = SklearnBoundaryModel.load(selected_boundary_model) if selected_boundary_model else WeightedFusionBoundary()
        probabilities = model.predict_proba(features)
        groups = group_pages(packet, [page.page_number for page in pages], probabilities, self.config.boundary.threshold, self.config.boundary.decision)
        selected_document_model = document_model or self.config.classification.model_path
        classifier, classifier_status = build_document_classifier(selected_document_model, self.config.classification.min_confidence, self.config.classification.model_kind)
        if classifier_status.get("warning"):
            print(f"[warning] {classifier_status['warning']}")
        by_page = {page.page_number: page for page in pages}
        # Layout descriptors come from the same pages as the text, so a layout-aware classifier
        # sees geometry built by the same code path at training and inference.
        group_texts = ["\n".join(by_page[number].text for number in group.pages) for group in groups]
        group_layouts = [shape_vector(document_shape_features([by_page[number] for number in group.pages]))
                         for group in groups]
        predictions = classifier.predict(group_texts, group_layouts)
        for group, prediction in zip(groups, predictions):
            group.doc_type, group.classification_confidence = prediction.label, prediction.confidence
            group.classification_source, group.classification_alternatives = prediction.source, prediction.alternatives
        structured = [structure_document(group, pages, source.name, str(source)) for group in groups]
        chunks = [chunk for document in structured for chunk in chunk_document(document, self.config.chunking)]
        dump_json([asdict(page) for page in pages], target / "pages.json")
        dump_json({"packet_id": packet, "documents": [asdict(group) for group in groups]}, target / "stage1.json")
        dump_json([asdict(document) for document in structured], target / "structured_documents.json")
        dump_json([asdict(chunk) for chunk in chunks], target / "chunks.json")
        markdown_dir = target / "markdown"; markdown_dir.mkdir(parents=True, exist_ok=True)
        for document in structured:
            (markdown_dir / f"{document.doc_id}.md").write_text(render_markdown(document), encoding="utf-8")
        HybridRetriever(chunks, self.config.retrieval).save(str(target / "index"))
        # The raw per-pair probabilities are the actual Stage 1 decision surface. Groups only
        # retain the within-document ones, so the values at the splits would otherwise be lost.
        result: dict[str, Any] = {"packet_id": packet, "documents": [asdict(group) for group in groups], "chunk_count": len(chunks), "output_dir": str(target), "boundary_probabilities": [float(probability) for probability in probabilities], "model_status": {"boundary": "trained" if selected_boundary_model else "weighted_fusion_fallback", "document_classifier": classifier_status}}
        # Additive: the Stage 2 structure and chunks are returned alongside the original keys so
        # callers do not have to read the JSON written to disk to see the structured output.
        if include_structure:
            result["structured_documents"] = [asdict(document) for document in structured]
        if include_chunks:
            result["chunks"] = [asdict(chunk) for chunk in chunks]
        return result

    def _retriever_for(self, processed_dir: str | Path) -> HybridRetriever:
        """Build (and cache) the retriever for a processed directory.

        Rebuilding on every call refits the dense encoder each time -- roughly 7s per query for
        the SVD encoder and over a minute for the transformer, which would dominate retrieval
        latency entirely. The cache is keyed on the chunk file's mtime so a reprocessed packet
        transparently invalidates it.
        """
        path = Path(processed_dir) / "chunks.json"
        key = (str(path.resolve()), path.stat().st_mtime_ns)
        cached = self._retriever_cache.get(key)
        if cached is None:
            chunks = [Chunk(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
            cached = HybridRetriever(chunks, self.config.retrieval)
            self._retriever_cache = {key: cached}  # single-entry cache; packets are large
        return cached

    def retrieve_from_dir(self, query: str, processed_dir: str | Path, top_k: int | None = None, query_aware: bool = False) -> list[dict[str, Any]]:
        return [asdict(result) for result in self._retriever_for(processed_dir).retrieve(query, top_k, query_aware)]

    def context_from_dir(self, query: str, processed_dir: str | Path, top_k: int | None = None,
                         query_aware: bool = False, token_budget: int | None = None) -> dict[str, Any]:
        """Retrieve and assemble a grounded, citation-carrying context block.

        This is the hand-off a RAG generator would consume. No text is generated here: the brief
        requires evidence retrieval rather than answers, so the stage stops at the point where a
        model would be called.
        """
        results = self._retriever_for(processed_dir).retrieve(query, top_k, query_aware)
        assembled = assemble_context(results, token_budget or self.config.retrieval.context_token_budget)
        return {"query": query, **asdict(assembled)}
