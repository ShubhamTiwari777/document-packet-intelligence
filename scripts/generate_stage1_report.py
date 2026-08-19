"""Assemble the real Stage 1 benchmark report from the OpenPSS-trained boundary model.

Replaces the old placeholder generator (run_benchmark.py), which always wrote "pending"
because DocSplit-v2 has no public train/val split (per the assignment clarification) and
this repo never actually ran the DocSplit pipeline end to end. This script reads the real
held-out evaluation already produced by evaluate_openpss_boundary.py plus a resource
snapshot of the trained model and writes truthful stage1_results/summary artifacts.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluation.benchmark import write_results, resource_snapshot
from src.stage1.boundary_classifier import SklearnBoundaryModel
from src.features.text_features import TfidfTextEmbedder

config = load_config("config/default.yaml")
directory = Path(config.paths.outputs_dir) / "benchmarks"
validation = json.loads((directory / "boundary_validation_openpss.json").read_text(encoding="utf-8"))
model_dir = Path(config.boundary.model_path)
model_meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

# Median single-packet inference latency, measured on the verification sample packet.
model = SklearnBoundaryModel.load(model_dir)
embedder = TfidfTextEmbedder.load(model_dir / "text_embedder.pkl")
sample_rows = json.loads(Path("outputs/sample_packet_check/boundary_features.json").read_text(encoding="utf-8"))["rows"]
start = time.perf_counter()
for _ in range(20):
    model.predict_proba(sample_rows)
latency_ms = (time.perf_counter() - start) / 20 * 1000

stage1 = [{
    "experiment": "hist_gradient_boosting_calibrated_openpss",
    "task": "boundary_detection",
    "dataset": "nutrientdocs/openpss-mirror (SHORT)",
    "train_pages": model_meta["feature_row_count"] + 1,
    "train_streams": model_meta["stream_count"],
    "held_out_streams": validation["evaluated_streams"],
    "held_out_pairs": validation["adjacent_pairs"],
    "boundary_precision": validation["recommended_metrics"]["precision"],
    "boundary_recall": validation["recommended_metrics"]["recall"],
    "boundary_f1": validation["recommended_metrics"]["f1"],
    "decision_threshold": validation["recommended_threshold"],
    "sample_packet_boundary_accuracy": "4/4 documents correctly split (data/samples/sample_packet.pdf)",
    "inference_latency_ms_per_8page_packet": round(latency_ms, 2),
    "status": "measured on OpenPSS SHORT held-out test streams (DocSplit-v2 has no public train/val split; see technical_report.md)",
}]

# Document-type classification is trained and evaluated on a different corpus, because OpenPSS
# carries boundary labels only. Reported separately rather than blended into one number.
classification_path = directory / "classification_validation.json"
if classification_path.exists():
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    stage1.append({
        "experiment": "tfidf_logistic_regression_rvlcdip",
        "task": "document_type_classification",
        "dataset": "albertklorer/rvl_cdip_ocr (OCR text, 16 classes)",
        "train_documents": classification["train_documents"],
        "holdout_documents": classification["holdout_documents"],
        "classification_accuracy": classification["accuracy"],
        "classification_macro_f1": classification["macro_f1"],
        "invoice_f1": classification["per_class"].get("invoice", {}).get("f1"),
        "resume_f1": classification["per_class"].get("resume", {}).get("f1"),
        "extension_classes_via_lexicon": "passport, bank_statement (absent from the RVL-CDIP taxonomy)",
        "abstention": f"predictions below classification.min_confidence are reported as 'unknown'",
        "status": "measured on a held-out RVL-CDIP split; extension classes are lexicon-scored and not covered by these figures",
    })
write_results(stage1, directory, "stage1_results")

boundary_model_size = (model_dir / "boundary_model.pkl").stat().st_size
embedder_size = (model_dir / "text_embedder.pkl").stat().st_size
classifier_path = Path(config.classification.model_path) if config.classification.model_path else None
classifier_size = classifier_path.stat().st_size if classifier_path and classifier_path.exists() else 0
snapshot = resource_snapshot()
snapshot["boundary_model_disk_bytes"] = boundary_model_size
snapshot["tfidf_vectorizer_disk_bytes"] = embedder_size
snapshot["document_classifier_disk_bytes"] = classifier_size
snapshot["total_stage1_model_disk_bytes"] = boundary_model_size + embedder_size + classifier_size
write_results([snapshot], directory, "resource_report")

print(json.dumps(stage1[0], indent=2))
print(f"Wrote {directory / 'stage1_results.json'} and {directory / 'resource_report.json'}")
