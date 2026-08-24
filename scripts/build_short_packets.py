"""Rebuild OpenPSS into short, boundary-dense packets that match the target regime.

The shipped model is trained on OpenPSS streams: a median of 21 pages with 11% of adjacent pairs
being boundaries. The DocSplit target is a median of 4 pages with 72% boundaries. That inversion
means the model learned "a boundary needs strong evidence" when the informative rare event in the
target is a *continuation*, and no threshold repairs it.

An earlier attempt to fix this with RVL-CDIP failed because its rows are independent single-page
documents, so grouping them fabricated continuity that did not exist. OpenPSS does not have that
problem: its boundary labels delimit real multi-page documents, 67% of which are 1-3 pages. This
script cuts those documents out and reassembles them into short packets, so both boundaries and
continuations are genuine while the density matches deployment.

The page content is untouched real data; only the packet composition changes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/raw/openpss/train/manifest.json")
parser.add_argument("--output", default="data/raw/openpss/short_packets/manifest.json")
parser.add_argument("--packets", type=int, default=900)
parser.add_argument("--max-document-pages", type=int, default=3, help="only reuse short documents")
parser.add_argument("--min-documents", type=int, default=2)
parser.add_argument("--max-documents", type=int, default=6)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

source = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

# Cut each stream into its constituent documents using the boundary labels.
documents: list[list[dict]] = []
for stream in source["streams"]:
    pages, labels = stream["pages"], stream["boundary_labels"]
    current = [pages[0]]
    for index, page in enumerate(pages[1:]):
        if labels[index] == 1:
            documents.append(current)
            current = [page]
        else:
            current.append(page)
    documents.append(current)

usable = [document for document in documents if 1 <= len(document) <= args.max_document_pages]
rng = random.Random(args.seed)
rng.shuffle(usable)
print(f"documents available: {len(documents)} total, {len(usable)} of 1-{args.max_document_pages} pages")

# Weight sampling toward 1-2 page documents so packet density approaches the target's ~72%.
streams: list[dict] = []
cursor = 0
for index in range(args.packets):
    count = rng.randint(args.min_documents, args.max_documents)
    if cursor + count > len(usable):
        break
    chosen = usable[cursor:cursor + count]
    cursor += count
    pages: list[dict] = []
    labels: list[int] = []
    for position, document in enumerate(chosen):
        for offset, page in enumerate(document):
            if pages:  # every pair after the first page gets a label
                labels.append(1 if offset == 0 else 0)
            pages.append({**page, "position": len(pages) + 1})
    if len(pages) < 2:
        continue
    streams.append({
        "stream_id": f"short_{index:05d}", "page_count": len(pages),
        "pages": pages, "boundary_labels": labels,
    })

pairs = sum(len(s["boundary_labels"]) for s in streams)
positives = sum(sum(s["boundary_labels"]) for s in streams)
manifest = {
    "dataset": "openpss-short-packets (rebuilt from nutrientdocs/openpss-mirror)",
    "config": source.get("config"), "split": "train",
    "stream_count": len(streams), "page_count": sum(s["page_count"] for s in streams),
    "positive_boundaries": positives, "streams": streams,
}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest), encoding="utf-8")

median_pages = sorted(s["page_count"] for s in streams)[len(streams) // 2] if streams else 0
print(f"packets {len(streams)}  pages {manifest['page_count']}  pairs {pairs}  "
      f"positives {positives}  density {positives / max(pairs, 1):.1%}  median {median_pages} pages/packet")
print(f"-> {output}")
