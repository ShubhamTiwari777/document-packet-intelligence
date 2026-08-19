from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.error_analysis import save_error_analysis

save_error_analysis({"status": "pending: run evaluated predictions with ground truth before collecting representative false positives/negatives", "stage1": [], "stage2": [], "stage3": []}, "outputs/error_analysis")
print("Wrote outputs/error_analysis/error_analysis.json")
