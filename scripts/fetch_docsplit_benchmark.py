"""Fetch the public DocSplit benchmark (evaluation only -- never used for training).

`nutrientdocs/doc-split-benchmark` ships a test split and nothing else, matching the guidance
that it is an evaluation benchmark rather than a training corpus. Scoring on it measures the
system against the task the assignment actually targets; training on it would be exactly the
benchmark-fitting the brief prohibits.

Its schema mirrors OpenPSS with different field names (boundary/page_text rather than
label/text), so it is normalised into the same manifest structure the OpenPSS tooling consumes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.openpss_dataset import _get_with_backoff, group_into_streams, materialize_stream

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="data/raw/docsplit_benchmark")
parser.add_argument("--config", default="our200")
parser.add_argument("--split", default="test")
parser.add_argument("--max-rows", type=int, default=1000)
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()

DATASET = "nutrientdocs%2Fdoc-split-benchmark"
rows: list[dict] = []
offset = 0
while offset < args.max_rows:
    length = min(100, args.max_rows - offset)
    url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}"
           f"&config={args.config}&split={args.split}&offset={offset}&length={length}")
    payload = json.loads(_get_with_backoff(url, retries=8, base_delay=3.0))
    batch = payload["rows"]
    if not batch:
        break
    for item in batch:
        row = item["row"]
        # Normalise onto the OpenPSS field names so existing tooling applies unchanged.
        rows.append({"stream_id": row["stream_id"], "position": row["position"],
                     "text": row.get("page_text") or "", "label": int(row.get("boundary", 0)),
                     "image": row["image"]})
    offset += len(batch)
    if len(batch) < length:
        break

output = Path(args.output)
images = output / "images"
streams = [materialize_stream(group, images, args.workers) for group in group_into_streams(iter(rows))]
manifest = {
    "dataset": "nutrientdocs/doc-split-benchmark", "config": args.config, "split": args.split,
    "stream_count": len(streams), "page_count": sum(s["page_count"] for s in streams),
    "positive_boundaries": sum(sum(s["boundary_labels"]) for s in streams), "streams": streams,
}
output.mkdir(parents=True, exist_ok=True)
(output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
pairs = sum(len(s["boundary_labels"]) for s in streams)
print(f"streams {len(streams)}  pages {manifest['page_count']}  pairs {pairs}  "
      f"positives {manifest['positive_boundaries']}  base rate {manifest['positive_boundaries']/max(pairs,1):.3%}")
