"""OpenPSS (nutrientdocs/openpss-mirror) adapter.

Fetches labeled page streams via the public HF datasets-server rows API instead of the
parquet files directly: this repo's Python (3.14) has no prebuilt `pyarrow`/`datasets`
wheel available and building them from source requires a local C++ toolchain, so the
REST API is the lightest-weight path to the same rows (stream_id, position, image, text,
label) without adding heavy/binary dependencies.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator
import json
import time
import urllib.error
import urllib.request

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
PAGE_REQUEST_DELAY = 0.4


def _get_with_backoff(url: str, retries: int, base_delay: float) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if attempt == retries - 1:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = float(retry_after) if retry_after else base_delay * (2 ** attempt)
            time.sleep(min(delay, 60.0))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (attempt + 1))
    raise RuntimeError("unreachable")  # pragma: no cover


def iter_rows(dataset: str, config: str, split: str, max_rows: int | None = None, retries: int = 8) -> Iterator[dict[str, Any]]:
    """Yield raw rows in dataset order, paging through the datasets-server API."""
    offset = 0
    while max_rows is None or offset < max_rows:
        length = PAGE_SIZE if max_rows is None else min(PAGE_SIZE, max_rows - offset)
        url = f"{ROWS_ENDPOINT}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
        payload = json.loads(_get_with_backoff(url, retries, base_delay=3.0))
        rows = payload["rows"]
        if not rows:
            return
        for item in rows:
            yield item["row"]
        offset += len(rows)
        if len(rows) < length:
            return
        time.sleep(PAGE_REQUEST_DELAY)


def group_into_streams(rows: Iterator[dict[str, Any]], max_streams: int | None = None) -> Iterator[list[dict[str, Any]]]:
    """Group consecutive same-stream_id rows; never truncates a stream mid-way."""
    current_id: str | None = None
    current: list[dict[str, Any]] = []
    emitted = 0
    for row in rows:
        if row["stream_id"] != current_id:
            if current:
                emitted += 1
                yield current
                if max_streams is not None and emitted >= max_streams:
                    return
            current_id, current = row["stream_id"], []
        current.append(row)
    if current:
        yield current


def _download_image(url: str, destination: Path, retries: int = 8) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_get_with_backoff(url, retries, base_delay=1.5))


def materialize_stream(stream_rows: list[dict[str, Any]], images_dir: Path, workers: int = 16) -> dict[str, Any]:
    """Download each page's image and return a stream manifest entry."""
    stream_id = stream_rows[0]["stream_id"]
    safe_id = stream_id.replace("/", "_")
    stream_dir = images_dir / safe_id
    tasks = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in stream_rows:
            path = stream_dir / f"page_{row['position']:04d}.jpg"
            tasks[pool.submit(_download_image, row["image"]["src"], path)] = (row, path)
        pages: dict[int, dict[str, Any]] = {}
        for future in as_completed(tasks):
            row, path = tasks[future]
            future.result()
            pages[row["position"]] = {
                "position": row["position"],
                "image_path": str(path),
                "width": row["image"]["width"],
                "height": row["image"]["height"],
                "text": row["text"] or "",
                "label": int(row["label"]),
            }
    ordered = [pages[p] for p in sorted(pages)]
    boundary_labels = [page["label"] for page in ordered[1:]]
    return {"stream_id": stream_id, "page_count": len(ordered), "boundary_labels": boundary_labels, "pages": ordered}


def pages_from_stream(stream: dict[str, Any], synthesize_blocks_from_text: bool = True) -> list:
    """Build PageRepresentations for one manifest stream.

    OpenPSS supplies OCR text and a page image but no blocks or fonts, so eight of the fourteen
    boundary features -- including header_similarity and footer_similarity, two of the strongest
    published page-stream-segmentation signals -- were constant zero across the entire training
    set and could carry no information. Reconstructing pseudo-blocks from the text (the same
    routine Stage 2 uses for OCR-only input) gives those features real values at both training
    and inference time.
    """
    from dataclasses import replace as _replace
    from src.domain import PageRepresentation
    from src.stage2.text_structure import synthesize_blocks

    pages = [
        PageRepresentation(
            page_number=page["position"], text=page["text"], blocks=[], fonts=[],
            width=float(page["width"]), height=float(page["height"]), image_path=page["image_path"],
        )
        for page in stream["pages"]
    ]
    if not synthesize_blocks_from_text:
        return pages
    return [_replace(page, blocks=synthesize_blocks(page)) for page in pages]


def fetch_openpss_manifest(
    output_dir: str | Path,
    config: str = "SHORT",
    split: str = "train",
    max_rows: int | None = 8000,
    max_streams: int | None = None,
    workers: int = 16,
    progress: bool = True,
) -> dict[str, Any]:
    """Download an OpenPSS split into a local manifest + page images. Resumable: existing images are skipped."""
    output = Path(output_dir)
    images_dir = output / "images"
    manifest_path = output / "manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    streams: list[dict[str, Any]] = []
    rows = iter_rows("nutrientdocs%2Fopenpss-mirror", config, split, max_rows)

    def _snapshot() -> dict[str, Any]:
        return {
            "dataset": "nutrientdocs/openpss-mirror", "config": config, "split": split,
            "stream_count": len(streams), "page_count": sum(s["page_count"] for s in streams),
            "positive_boundaries": sum(sum(s["boundary_labels"]) for s in streams),
            "streams": streams,
        }

    for index, stream_rows in enumerate(group_into_streams(rows, max_streams), 1):
        streams.append(materialize_stream(stream_rows, images_dir, workers))
        if index % 5 == 0:
            manifest_path.write_text(json.dumps(_snapshot(), indent=2), encoding="utf-8")
        if progress and (index % 10 == 0):
            total_pages = sum(s["page_count"] for s in streams)
            print(f"  fetched {index} streams / {total_pages} pages so far")
    manifest = _snapshot()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
