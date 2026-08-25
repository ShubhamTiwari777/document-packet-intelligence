"""Fetch OpenPSS page text and boundary labels without downloading page images.

The cross-encoder reads only the text at the page seam, so the image download that dominates
`fetch_openpss.py` is pure cost for it: page images are roughly two orders of magnitude larger
than the text and are needed only by `visual_delta` in the feature model. Skipping them makes it
practical to pull the whole split rather than a 16k-row slice, which is the bottleneck the
cross-encoder experiment ran into (1,600 training examples for a 135M-parameter model).

Emits the same manifest shape as the image fetcher, with `image_path` set to null.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.openpss_dataset import _get_with_backoff

DATASET = "nutrientdocs%2Fopenpss-mirror"
PAGE_SIZE = 100

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="SHORT", choices=["SHORT", "LONG"])
parser.add_argument("--split", default="train")
parser.add_argument("--output", required=True)
parser.add_argument("--max-rows", type=int, default=40000)
parser.add_argument("--start", type=int, default=0, help="row offset, to extend an earlier pull")
args = parser.parse_args()

rows: list[dict] = []
offset = args.start
while offset < args.start + args.max_rows:
    length = min(PAGE_SIZE, args.start + args.max_rows - offset)
    url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}"
           f"&config={args.config}&split={args.split}&offset={offset}&length={length}")
    payload = json.loads(_get_with_backoff(url, retries=8, base_delay=2.0))
    batch = payload["rows"]
    if not batch:
        break
    for item in batch:
        row = item["row"]
        rows.append({"stream_id": row["stream_id"], "position": row["position"],
                     "text": row.get("text") or "", "label": int(row.get("label", 0)),
                     "width": row["image"].get("width", 224) if isinstance(row.get("image"), dict) else 224,
                     "height": row["image"].get("height", 224) if isinstance(row.get("image"), dict) else 224})
    offset += len(batch)
    if offset % 2000 == 0:
        print(f"  {offset - args.start} rows", flush=True)
    if len(batch) < length:
        break

# Group consecutive rows by stream, preserving position order.
streams: list[dict] = []
current: list[dict] = []
for row in rows:
    if current and row["stream_id"] != current[-1]["stream_id"]:
        streams.append(current)
        current = []
    current.append(row)
if current:
    streams.append(current)

built: list[dict] = []
for group in streams:
    group.sort(key=lambda r: r["position"])
    if len(group) < 2:
        continue
    pages = [{"position": index + 1, "text": row["text"], "width": float(row["width"]),
              "height": float(row["height"]), "image_path": None}
             for index, row in enumerate(group)]
    # label[i] == 1 marks the first page of a document; the pair label is the right page's flag.
    labels = [int(row["label"]) for row in group[1:]]
    built.append({"stream_id": group[0]["stream_id"], "page_count": len(pages),
                  "pages": pages, "boundary_labels": labels})

pairs = sum(len(s["boundary_labels"]) for s in built)
positives = sum(sum(s["boundary_labels"]) for s in built)
manifest = {"dataset": "nutrientdocs/openpss-mirror (text only)", "config": args.config,
            "split": args.split, "stream_count": len(built),
            "page_count": sum(s["page_count"] for s in built),
            "positive_boundaries": positives, "streams": built}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest), encoding="utf-8")
print(f"streams {len(built)}  pages {manifest['page_count']}  pairs {pairs}  "
      f"positives {positives}  density {positives / max(pairs, 1):.1%}")
print(f"-> {output}")
