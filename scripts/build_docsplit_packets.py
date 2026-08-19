"""Reconstruct trainable PDF packets from DocSplit labels and local source PDFs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import resolve_docsplit_pdf


def build_packet(packet: dict, raw_root: Path, output_pdf: Path) -> tuple[bool, str | None]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required. Activate .venv and run pip install -r requirements.txt") from exc
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    assembled = fitz.open()
    try:
        for document in packet["documents"]:
            for source in document["source_pages"]:
                source_pdf = resolve_docsplit_pdf(raw_root, document["doc_type"], source["original_doc_name"])
                if source_pdf is None:
                    return False, f"missing source PDF: {document['doc_type']}/{source['original_doc_name']}"
                with fitz.open(source_pdf) as original:
                    index = source["source_page"] - 1
                    if index < 0 or index >= len(original):
                        return False, f"missing source page {source['source_page']} in {source_pdf.name}"
                    assembled.insert_pdf(original, from_page=index, to_page=index)
        assembled.save(output_pdf, garbage=4, deflate=True)
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        assembled.close()


parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--raw-pdfs", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--limit", type=int, default=500, help="Maximum packets; use 0 for all")
args = parser.parse_args()

manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
selected = manifest if args.limit == 0 else manifest[:args.limit]
output = Path(args.output); packet_dir = output / "packets"; packet_dir.mkdir(parents=True, exist_ok=True)
built: list[dict] = []; skipped: list[dict] = []
for index, packet in enumerate(selected, 1):
    success, reason = build_packet(packet, Path(args.raw_pdfs), packet_dir / f"{packet['packet_id']}.pdf")
    if success:
        built.append({"packet_id": packet["packet_id"], "pdf_path": str(packet_dir / f"{packet['packet_id']}.pdf"), "boundary_labels": packet["boundary_labels"], "documents": packet["documents"]})
    else:
        (packet_dir / f"{packet['packet_id']}.pdf").unlink(missing_ok=True)
        skipped.append({"packet_id": packet["packet_id"], "reason": reason})
    if index % 50 == 0 or index == len(selected): print(f"Built {len(built)}/{index}; skipped {len(skipped)}")
(output / "packet_index.json").write_text(json.dumps({"built": built, "skipped": skipped}, indent=2), encoding="utf-8")
print(f"Packet index: {output / 'packet_index.json'}")
