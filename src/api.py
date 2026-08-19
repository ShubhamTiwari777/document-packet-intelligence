"""Thin FastAPI surface for the pipeline."""
from __future__ import annotations

from pathlib import Path
import tempfile

from src.config import AppConfig, load_config
from src.pipeline import DocumentPipeline

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile  # type: ignore
    from pydantic import BaseModel  # type: ignore
except ImportError:  # Keeps importing non-API pipeline modules lightweight.
    FastAPI = File = HTTPException = UploadFile = BaseModel = None  # type: ignore


if BaseModel is not None:
    class RetrieveRequest(BaseModel):
        query: str
        processed_dir: str
        top_k: int = 5
        query_aware: bool = False


def create_app(config_path: str | None = None):
    if FastAPI is None or BaseModel is None:
        raise RuntimeError("FastAPI dependencies are required to serve the API.")
    default_config = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    config = load_config(config_path or default_config)
    pipeline = DocumentPipeline(config)
    app = FastAPI(title="Document Packet Intelligence", version="0.1.0")
    app.state.config = config

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

    return app
