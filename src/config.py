"""Central configuration loading and path handling."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json


@dataclass
class PathsConfig:
    data_dir: str = "data"
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    cache_dir: str = "data/processed/cache"


@dataclass
class RuntimeConfig:
    device: str = "cpu"
    random_seed: int = 42
    render_dpi: int = 150
    max_render_pixels: int = 30_000_000
    log_level: str = "INFO"


@dataclass
class IngestionConfig:
    min_native_characters: int = 20
    min_text_density: float = 0.00005
    enable_ocr: bool = False
    ocr_engine: str = "tesseract"


@dataclass
class BoundaryConfig:
    threshold: float = 0.5
    model_kind: str = "hist_gradient_boosting"
    calibration: bool = False
    model_path: str | None = None


@dataclass
class ClassificationConfig:
    model_kind: str = "tfidf_logistic_regression"
    min_confidence: float = 0.35
    model_path: str | None = None


@dataclass
class ChunkingConfig:
    max_tokens: int = 256
    overlap_tokens: int = 32


@dataclass
class RetrievalConfig:
    top_k: int = 5
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rerank: bool = False
    # Dense encoder: "hashed" (bag-of-words baseline), "svd" (TF-IDF+LSA), or "transformer".
    encoder: str = "svd"
    embedding_model: str = "BAAI/bge-small-en-v1.5"


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown configuration key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(value, dict):
                raise ValueError(f"Expected mapping for configuration section: {key}")
            _merge(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config when PyYAML is installed; JSON is always supported."""
    config = AppConfig()
    if path is None:
        return config
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        values = json.loads(content)
    else:
        try:
            import yaml  # type: ignore
            values = yaml.safe_load(content) or {}
        except ImportError:
            # The checked-in default is intentionally a flat two-level YAML subset.
            # This keeps smoke tests runnable before optional dependencies are installed.
            values = _parse_simple_yaml(content)
    return _merge(config, values)


def _parse_simple_yaml(content: str) -> dict[str, Any]:
    """Parse mapping-only YAML used by config/default.yaml; install PyYAML for full YAML."""
    output: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw in content.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            key = line.rstrip(":").strip(); current = {}; output[key] = current; continue
        if current is None or ":" not in line:
            raise ValueError("Unsupported YAML syntax. Install PyYAML for full YAML support.")
        key, value = (part.strip() for part in line.split(":", 1))
        if value.lower() in {"true", "false"}: parsed: Any = value.lower() == "true"
        elif value.lower() in {"null", "none"}: parsed = None
        else:
            try: parsed = int(value)
            except ValueError:
                try: parsed = float(value)
                except ValueError: parsed = value.strip("\"'")
        current[key] = parsed
    return output
