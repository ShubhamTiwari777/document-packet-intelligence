"""Representative error records, selected from benchmark prediction artifacts."""
from __future__ import annotations

from pathlib import Path
import json


def collect_binary_errors(rows: list[dict], prediction_key: str, actual_key: str) -> dict[str, list[dict]]:
    return {"false_positives": [row for row in rows if row.get(prediction_key) == 1 and row.get(actual_key) == 0], "false_negatives": [row for row in rows if row.get(prediction_key) == 0 and row.get(actual_key) == 1]}


def save_error_analysis(analysis: dict, output_dir: str | Path) -> None:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    (target / "error_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
