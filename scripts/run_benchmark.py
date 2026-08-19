"""Create reproducible benchmark artifacts; reports pending values when labels are unavailable."""
from __future__ import annotations
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.evaluation.benchmark import write_results, resource_snapshot

config = load_config("config/default.yaml")
directory = Path(config.paths.outputs_dir) / "benchmarks"
pending = "pending: supply DocSplit-v2 local labels and retrieval judgments"
stage1 = [{"experiment": "text_only", "boundary_precision": None, "boundary_recall": None, "boundary_f1": None, "status": pending}, {"experiment": "text_visual", "boundary_precision": None, "boundary_recall": None, "boundary_f1": None, "status": pending}, {"experiment": "text_visual_heuristics_weighted", "boundary_precision": None, "boundary_recall": None, "boundary_f1": None, "status": pending}, {"experiment": "hist_gradient_boosting", "boundary_precision": None, "boundary_recall": None, "boundary_f1": None, "status": pending}]
stage2 = [{"experiment": "rule_based_structure", "text_extraction_coverage": None, "heading_f1": None, "table_f1": None, "status": pending}]
stage3 = [{"experiment": name, "recall@1": None, "recall@3": None, "recall@5": None, "mrr": None, "ndcg": None, "status": pending} for name in ("dense", "bm25", "fixed_rrf", "query_aware_rrf", "rrf_reranker")]
write_results(stage1, directory, "stage1_results"); write_results(stage2, directory, "stage2_results"); write_results(stage3, directory, "stage3_results")
write_results(stage1 + stage2 + stage3, directory, "summary"); write_results([resource_snapshot()], directory, "resource_report")
print(f"Wrote pending, non-fabricated benchmark artifacts to {directory}")
