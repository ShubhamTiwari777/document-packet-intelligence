"""FastAPI surface for the pipeline, plus the browser UI.

Two layers live here. The original endpoints (`/process`, `/retrieve`, `/context`) take and
return raw paths and are kept unchanged for scripts and tests. The `/api/*` layer added for the
UI is job-based instead: the client only ever sees an opaque job id, and every path is resolved
underneath the outputs directory, so a caller cannot walk the filesystem by sending `../`.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import re
import shutil
import tempfile
import time
import uuid

from src.config import AppConfig, load_config
from src.evaluation.packet_eval import evaluate_packet
from src.export import archive, slugify, split_packet
from src.pipeline import DocumentPipeline

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # type: ignore
    from pydantic import BaseModel  # type: ignore
except ImportError:  # Keeps importing non-API pipeline modules lightweight.
    FastAPI = File = Form = HTTPException = UploadFile = BaseModel = None  # type: ignore

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
SAMPLE_PDF = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_packet.pdf"
SAMPLE_TRUTH = Path(__file__).resolve().parents[1] / "data" / "samples" / "ground_truth.json"

# Held-out benchmark results for the shipped models, quoted in the UI so the numbers on screen
# are never mistaken for something measured on the user's own upload. Sources: technical report
# section 5.1 and the measurement comments in config/default.yaml.
BENCHMARKS: dict[str, Any] = {
    "boundary": {
        "label": "Document splitting (English model)",
        "metrics": [
            {"name": "Page grouping accuracy", "value": 0.968, "dataset": "TABME++ held-out test, 501 English packets"},
            {"name": "Boundary F1", "value": 0.942, "dataset": "TABME++ held-out test, 2,284 seams"},
            {"name": "Trivial baseline", "value": 0.720, "dataset": "always-split, 56% boundary density", "baseline": True},
            {"name": "Same model on Dutch", "value": 0.606, "dataset": "OpenPSS — wrong language, see note", "baseline": True},
        ],
    },
    "classification": {
        "label": "Document type",
        "metrics": [
            {"name": "Accuracy", "value": 0.842, "dataset": "5,417 held-out documents, 15 types"},
            {"name": "Macro F1", "value": 0.841, "dataset": "5,417 held-out documents, 15 types"},
        ],
    },
    "retrieval": {
        "label": "Evidence retrieval",
        "metrics": [
            {"name": "Recall@1", "value": 0.771, "dataset": "35 queries, 2.6k chunks"},
            {"name": "Recall@5", "value": 0.914, "dataset": "35 queries, 2.6k chunks"},
            {"name": "MRR", "value": 0.829, "dataset": "35 queries, 2.6k chunks"},
        ],
    },
    "note": ("Measured on held-out public benchmarks, not on your upload. Accuracy and precision "
             "for an uploaded packet can only be computed when ground truth is supplied. The "
             "splitter is language-specific: the shipped model is trained on English and its "
             "boundary recall falls to 0.34 on Dutch, where it silently merges documents instead "
             "of splitting them."),
}


if BaseModel is not None:
    class RetrieveRequest(BaseModel):
        query: str
        processed_dir: str
        top_k: int = 5
        query_aware: bool = False

    class ContextRequest(BaseModel):
        query: str
        processed_dir: str
        top_k: int = 8
        query_aware: bool = False
        token_budget: int | None = None

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 5
        query_aware: bool = False


def create_app(config_path: str | None = None):
    if FastAPI is None or BaseModel is None:
        raise RuntimeError("FastAPI dependencies are required to serve the API.")
    from fastapi.responses import FileResponse, JSONResponse  # type: ignore
    from fastapi.staticfiles import StaticFiles  # type: ignore

    default_config = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    config = load_config(config_path or default_config)
    pipeline = DocumentPipeline(config)
    app = FastAPI(title="Document Packet Intelligence", version="1.0.0")
    app.state.config = config

    jobs_root = Path(config.paths.outputs_dir) / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def job_directory(job_id: str) -> Path:
        """Resolve a job id to its directory, refusing anything that escapes the jobs root."""
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", job_id or ""):
            raise HTTPException(400, "Malformed job id.")
        directory = (jobs_root / job_id).resolve()
        if not str(directory).startswith(str(jobs_root.resolve())):
            raise HTTPException(400, "Malformed job id.")
        if not directory.is_dir():
            raise HTTPException(404, f"No such job: {job_id}")
        return directory

    def read_job(job_id: str) -> dict[str, Any]:
        payload = job_directory(job_id) / "job.json"
        if not payload.exists():
            raise HTTPException(404, f"Job {job_id} has no result recorded.")
        return json.loads(payload.read_text(encoding="utf-8"))

    def run_job(source_pdf: Path, display_name: str, truth: list[dict[str, Any]] | None) -> dict[str, Any]:
        """Process one packet end to end and persist everything the UI needs."""
        job_id = f"{slugify(Path(display_name).stem)}-{uuid.uuid4().hex[:8]}"
        directory = jobs_root / job_id
        directory.mkdir(parents=True, exist_ok=True)
        stored_pdf = directory / "source.pdf"
        if Path(source_pdf).resolve() != stored_pdf.resolve():
            shutil.copyfile(source_pdf, stored_pdf)

        started = time.perf_counter()
        result = pipeline.process(stored_pdf, directory, packet_id=Path(display_name).stem)
        elapsed = time.perf_counter() - started

        documents = result.get("documents", [])
        manifest = split_packet(stored_pdf, documents, directory / "split")
        archive(directory / "split", directory / "split.zip")

        pages = json.loads((directory / "pages.json").read_text(encoding="utf-8"))
        structured = {d["doc_id"]: d for d in result.get("structured_documents", [])}
        chunks = result.get("chunks", [])
        chunk_counts: dict[str, int] = {}
        for chunk in chunks:
            chunk_counts[chunk["doc_id"]] = chunk_counts.get(chunk["doc_id"], 0) + 1

        by_manifest = {entry["doc_id"]: entry for entry in manifest}
        enriched: list[dict[str, Any]] = []
        for position, document in enumerate(documents, start=1):
            entry = by_manifest.get(document["doc_id"], {})
            structure = structured.get(document["doc_id"], {})
            sections = structure.get("sections", [])
            element_types: dict[str, int] = {}
            for section in sections:
                for element in section.get("elements", []):
                    element_types[element["type"]] = element_types.get(element["type"], 0) + 1
            enriched.append({
                "index": entry.get("index", position),
                "doc_id": document["doc_id"],
                "doc_type": document.get("doc_type", "unknown"),
                "pages": document.get("pages", []),
                "page_count": len(document.get("pages", [])),
                "confidence": round(float(document.get("classification_confidence", 0.0)), 4),
                "classification_source": document.get("classification_source", "unknown"),
                "alternatives": document.get("classification_alternatives", [])[:3],
                "boundary_confidences": [round(float(v), 4) for v in document.get("boundary_confidences", [])],
                "section_count": len(sections),
                "chunk_count": chunk_counts.get(document["doc_id"], 0),
                "elements": element_types,
                "headings": [s.get("title", "") for s in sections if s.get("title")][:6],
                "filename": entry.get("filename"),
                "bytes": entry.get("bytes", 0),
                "download": f"/api/jobs/{job_id}/documents/{entry.get('index', position)}/pdf",
            })

        # Stage 1 works on adjacent page pairs, so expose that decision surface directly: for
        # each seam, the probability a new document starts there and whether it became a split.
        starts = {min(d["pages"]) for d in documents if d.get("pages")}
        probabilities = result.get("boundary_probabilities", [])
        page_pairs = [{
            "from": index + 1,
            "to": index + 2,
            "probability": round(float(probability), 4),
            "split": (index + 2) in starts,
        } for index, probability in enumerate(probabilities)]

        payload: dict[str, Any] = {
            "job_id": job_id,
            "filename": display_name,
            "page_count": len(pages),
            "page_pairs": page_pairs,
            "expected_boundaries": round(sum(probabilities), 4),
            "decision_rule": config.boundary.decision,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "elapsed_seconds": round(elapsed, 2),
            "documents": enriched,
            "archive": f"/api/jobs/{job_id}/archive",
            "source_pdf": f"/api/jobs/{job_id}/source.pdf",
            "model_status": result.get("model_status", {}),
            "evaluation": None,
        }
        if truth:
            payload["evaluation"] = evaluate_packet(documents, truth, len(pages))
        (directory / "job.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def parse_truth(raw: str | None) -> list[dict[str, Any]] | None:
        """Accept either a full ground-truth file or a bare list of documents."""
        if not raw or not raw.strip():
            return None
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as error:
            raise HTTPException(400, f"Ground truth is not valid JSON: {error}")
        documents = loaded.get("documents") if isinstance(loaded, dict) else loaded
        if not isinstance(documents, list):
            raise HTTPException(400, "Ground truth must be a list of documents, or an object with a 'documents' list.")
        for item in documents:
            if not isinstance(item, dict) or "pages" not in item:
                raise HTTPException(400, "Each ground-truth document needs a 'pages' list.")
        return documents

    # ------------------------------------------------------------------ original API
    @app.get("/health")
    def health() -> dict[str, str]:
        model_path = config.boundary.model_path
        return {"status": "ok", "boundary_model": "configured" if model_path and Path(model_path).is_dir() else "fallback"}

    @app.post("/process")
    async def process(file: UploadFile = File(...), include_structure: bool = True, include_chunks: bool = True):
        """Run the packet through Stage 1 + Stage 2.

        `include_structure` / `include_chunks` are opt-out query parameters: the structured
        sections and chunks are returned by default so the API response is self-describing,
        but a caller processing a very large packet can suppress them and read the JSON
        artifacts from `output_dir` instead.
        """
        if not file.filename or not file.filename.lower().endswith(".pdf"): raise HTTPException(400, "A PDF file is required.")
        output = Path(config.paths.outputs_dir) / Path(file.filename).stem
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
            temporary.write(await file.read()); source = temporary.name
        return pipeline.process(source, output, include_structure=include_structure, include_chunks=include_chunks)

    @app.post("/index")
    def index(processed_dir: str) -> dict[str, str]:
        if not (Path(processed_dir) / "chunks.json").exists(): raise HTTPException(404, "chunks.json not found")
        return {"status": "existing index is regenerated on process", "processed_dir": processed_dir}

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequest): return {"query": request.query, "results": pipeline.retrieve_from_dir(request.query, request.processed_dir, request.top_k, request.query_aware)}

    @app.post("/context")
    def context(request: ContextRequest):
        """Grounded, citation-carrying context for a retrieval-augmented generator.

        Deduplicated, budgeted to a context window, and annotated with document and page markers.
        No text is generated: the brief requires evidence retrieval rather than answers, so this
        is the hand-off point where a generator would be called.
        """
        return pipeline.context_from_dir(request.query, request.processed_dir, request.top_k,
                                         request.query_aware, request.token_budget)

    # ------------------------------------------------------------------ UI-facing API
    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        boundary_ready = bool(config.boundary.model_path and Path(config.boundary.model_path).is_dir())
        classifier_ready = bool(config.classification.model_path and Path(config.classification.model_path).exists())
        return {
            "status": "ok",
            "boundary_model": config.boundary.model_path if boundary_ready else "weighted-fusion fallback",
            "boundary_ready": boundary_ready,
            "classifier_model": config.classification.model_path if classifier_ready else "lexicon fallback",
            "classifier_ready": classifier_ready,
            "decision_rule": config.boundary.decision,
            "encoder": config.retrieval.encoder,
            "rerank": config.retrieval.rerank,
            "sample_available": SAMPLE_PDF.exists(),
        }

    @app.get("/api/benchmarks")
    def api_benchmarks() -> dict[str, Any]:
        return BENCHMARKS

    @app.post("/api/process")
    async def api_process(file: UploadFile = File(...), ground_truth: str | None = Form(default=None)):
        """Upload a packet, optionally with ground truth, and get the full result."""
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "A PDF file is required.")
        truth = parse_truth(ground_truth)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
            temporary.write(await file.read())
            staged = Path(temporary.name)
        try:
            return run_job(staged, file.filename, truth)
        finally:
            staged.unlink(missing_ok=True)

    @app.post("/api/process-sample")
    def api_process_sample():
        """Run the bundled 9-page sample packet, scored against its shipped ground truth."""
        if not SAMPLE_PDF.exists():
            raise HTTPException(404, "Sample packet not found. Run scripts/generate_sample_packet.py first.")
        truth = None
        if SAMPLE_TRUTH.exists():
            truth = json.loads(SAMPLE_TRUTH.read_text(encoding="utf-8")).get("documents")
        return run_job(SAMPLE_PDF, SAMPLE_PDF.name, truth)

    @app.get("/api/jobs")
    def api_jobs() -> dict[str, Any]:
        listing = []
        for payload in sorted(jobs_root.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:25]:
            data = json.loads(payload.read_text(encoding="utf-8"))
            listing.append({key: data.get(key) for key in
                            ("job_id", "filename", "page_count", "document_count", "elapsed_seconds")})
        return {"jobs": listing}

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        return read_job(job_id)

    @app.get("/api/jobs/{job_id}/documents/{index}/pdf")
    def api_document_pdf(job_id: str, index: int):
        directory = job_directory(job_id)
        for document in read_job(job_id)["documents"]:
            if document["index"] == index and document.get("filename"):
                path = directory / "split" / document["filename"]
                if path.exists():
                    return FileResponse(path, media_type="application/pdf", filename=document["filename"])
        raise HTTPException(404, f"Document {index} not found in job {job_id}.")

    @app.get("/api/jobs/{job_id}/archive")
    def api_archive(job_id: str):
        path = job_directory(job_id) / "split.zip"
        if not path.exists():
            raise HTTPException(404, "Archive not available for this job.")
        return FileResponse(path, media_type="application/zip", filename=f"{job_id}-split.zip")

    @app.get("/api/jobs/{job_id}/source.pdf")
    def api_source(job_id: str):
        path = job_directory(job_id) / "source.pdf"
        if not path.exists():
            raise HTTPException(404, "Source PDF not available.")
        return FileResponse(path, media_type="application/pdf")

    @app.get("/api/jobs/{job_id}/pages/{page}")
    def api_page_image(job_id: str, page: int):
        path = job_directory(job_id) / "rendered_pages" / f"page_{page:04d}.png"
        if not path.exists():
            raise HTTPException(404, f"No rendered image for page {page}.")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/jobs/{job_id}/search")
    def api_search(job_id: str, request: SearchRequest):
        directory = job_directory(job_id)
        if not (directory / "chunks.json").exists():
            raise HTTPException(404, "This job has no searchable index.")
        if not request.query.strip():
            raise HTTPException(400, "A query is required.")
        results = pipeline.retrieve_from_dir(request.query, directory, request.top_k, request.query_aware)
        return {"query": request.query, "results": results}

    # ------------------------------------------------------------------ UI
    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def ui():
            return FileResponse(WEB_DIR / "index.html")

    return app
