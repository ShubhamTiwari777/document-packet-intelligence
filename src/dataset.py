"""DocSplit-v2 adapter with an intentionally explicit local-data contract."""
from __future__ import annotations

from pathlib import Path
import csv
import json


def discover_packets(dataset_dir: str | Path) -> list[Path]:
    return sorted(Path(dataset_dir).rglob("*.pdf"))


def load_ground_truth(path: str | Path) -> list[dict]:
    """Load normalized JSON labels: packet_id, left_page, right_page, is_boundary, doc_type."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_ground_truth(rows: list[dict]) -> None:
    required = {"packet_id", "left_page", "right_page", "is_boundary"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing: raise ValueError(f"Ground truth row {index} is missing: {sorted(missing)}")


DOCSPLIT_COLUMNS = {"doc_type", "original_doc_name", "parent_doc_name", "local_doc_id", "page", "image_path", "text_path", "group_id", "local_doc_id_page_ordinal"}


def docsplit_csv_to_manifest(csv_path: str | Path) -> list[dict]:
    """Convert DocSplit's flat page CSV into packet/document/boundary labels.

    Each adjacent pair gets `is_boundary=1` when its `local_doc_id` changes.
    The output intentionally contains no inferred values: it is a normalized view
    of the publisher's training labels.
    """
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = DOCSPLIT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Not a supported DocSplit CSV; missing columns: {sorted(missing)}")
        rows = list(reader)
    packets: dict[str, list[dict]] = {}
    for row in rows:
        packets.setdefault(row["parent_doc_name"], []).append(row)
    manifest: list[dict] = []
    for packet_id, packet_rows in sorted(packets.items()):
        packet_rows.sort(key=lambda row: int(row["page"]))
        expected = list(range(1, len(packet_rows) + 1))
        pages = [int(row["page"]) for row in packet_rows]
        if pages != expected:
            raise ValueError(f"Packet {packet_id} page values are not a contiguous 1-based sequence.")
        documents: list[dict] = []
        for row in packet_rows:
            if not documents or documents[-1]["local_doc_id"] != row["local_doc_id"]:
                documents.append({"local_doc_id": row["local_doc_id"], "doc_type": row["doc_type"], "group_id": row["group_id"], "pages": [], "source_pages": []})
            documents[-1]["pages"].append(int(row["page"]))
            documents[-1]["source_pages"].append({"original_doc_name": row["original_doc_name"], "image_path": row["image_path"], "text_path": row["text_path"], "source_page": int(row["local_doc_id_page_ordinal"])})
        boundaries = [int(left["local_doc_id"] != right["local_doc_id"]) for left, right in zip(packet_rows, packet_rows[1:])]
        manifest.append({"packet_id": packet_id, "page_count": len(packet_rows), "boundary_labels": boundaries, "documents": documents})
    return manifest


def resolve_docsplit_pdf(raw_root: str | Path, doc_type: str, original_doc_name: str) -> Path | None:
    """Resolve a DocSplit source name against the extracted RVL-CDIP-N-MP tree."""
    folder = Path(raw_root) / doc_type
    names = [original_doc_name]
    if not original_doc_name.lower().endswith(".pdf"):
        names.append(f"{original_doc_name}.pdf")
    for name in names:
        candidate = folder / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None
