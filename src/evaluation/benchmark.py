"""Artifact writer and resource measurement for all benchmark scripts."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import csv
import json
import time


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    result: dict[str, float] = {}; start = time.perf_counter()
    yield result
    result["elapsed_seconds"] = time.perf_counter() - start


def write_results(rows: list[dict[str, object]], directory: str | Path, stem: str) -> None:
    target = Path(directory); target.mkdir(parents=True, exist_ok=True)
    (target / f"{stem}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    headers = sorted({key for row in rows for key in row})
    with (target / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers); writer.writeheader(); writer.writerows(rows)


def resource_snapshot(model_path: str | Path | None = None) -> dict[str, float | None]:
    disk_size = Path(model_path).stat().st_size if model_path and Path(model_path).exists() and Path(model_path).is_file() else None
    try:
        import psutil  # type: ignore
        ram = psutil.Process().memory_info().rss
    except ImportError:
        ram = None
    return {"process_rss_bytes": ram, "model_disk_bytes": disk_size}
