"""Fine-tune a cross-encoder to decide whether two adjacent pages continue or start a document.

Every prior Stage 1 attempt failed the same way: four feature sets and four learners all landed on
one precision/recall curve, and on short packets the system stayed below a trivial "split
everything" baseline. The diagnosis was that the informative rare event there is a *continuation*,
and deciding whether page i+1 continues page i needs to model text flowing across a page break --
which lexical overlap features cannot represent.

A cross-encoder attends jointly over both pages, so it can represent that relation directly.

Two design choices matter:

* **Only the boundary region is fed in.** A page can be thousands of tokens, but the evidence for
  continuation lives at the seam: the end of page i and the start of page i+1. Truncating to that
  window both fits the context limit and removes distracting body text.
* **The backbone is multilingual.** OpenPSS and the DocSplit benchmark are both Dutch, so an
  English-only encoder would be modelling the wrong language.

Output probabilities feed the existing `decide_boundaries` expected-count rule unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="data/raw/openpss/short_packets/manifest.json")
parser.add_argument("--output", default="models/boundary_cross_encoder")
parser.add_argument("--model", default="distilbert-base-multilingual-cased")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--max-length", type=int, default=256)
parser.add_argument("--window-words", type=int, default=110, help="words taken from each side of the seam")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def seam(left_text: str, right_text: str, words: int) -> tuple[str, str]:
    """Tail of the left page, head of the right page -- the region a boundary shows up in."""
    left = " ".join(left_text.split()[-words:])
    right = " ".join(right_text.split()[:words])
    return left or "[leeg]", right or "[leeg]"


def build_examples(manifest_path: str, window: int) -> list[dict]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    examples: list[dict] = []
    for stream in manifest["streams"]:
        pages, labels = stream["pages"], stream["boundary_labels"]
        if len(labels) != len(pages) - 1:
            continue
        for index, label in enumerate(labels):
            left, right = seam(pages[index]["text"], pages[index + 1]["text"], window)
            examples.append({"left": left, "right": right, "label": int(label),
                             "stream_id": stream["stream_id"]})
    return examples


examples = build_examples(args.manifest, args.window_words)
rng = random.Random(args.seed)
# Split by stream so pages of one packet never straddle train and validation.
streams = sorted({example["stream_id"] for example in examples})
rng.shuffle(streams)
holdout = set(streams[: max(1, len(streams) // 5)])
train = [e for e in examples if e["stream_id"] not in holdout]
validation = [e for e in examples if e["stream_id"] in holdout]
print(f"examples {len(examples)}  train {len(train)}  validation {len(validation)}  "
      f"positives {sum(e['label'] for e in train)}/{len(train)}")

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.manual_seed(args.seed)
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
model.train()


class PairDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def collate(batch):
    encoded = tokenizer([b["left"] for b in batch], [b["right"] for b in batch],
                        truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
    encoded["labels"] = torch.tensor([b["label"] for b in batch])
    return encoded


loader = DataLoader(PairDataset(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
# Class weights: the packet corpus is boundary-heavy, so an unweighted loss drifts toward always
# predicting a boundary -- the exact trivial behaviour this model exists to beat.
positive = sum(e["label"] for e in train)
weights = torch.tensor([len(train) / (2 * (len(train) - positive)), len(train) / (2 * positive)], dtype=torch.float)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

total = args.epochs * len(loader)
step = 0
for epoch in range(args.epochs):
    for batch in loader:
        labels = batch.pop("labels")
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        if step % 25 == 0 or step == total:
            print(f"  step {step}/{total}  loss {loss.item():.4f}", flush=True)

model.eval()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
model.save_pretrained(output)
tokenizer.save_pretrained(output)
(output / "metadata.json").write_text(json.dumps({
    "backbone": args.model, "manifest": args.manifest, "epochs": args.epochs,
    "window_words": args.window_words, "max_length": args.max_length,
    "train_pairs": len(train), "validation_pairs": len(validation),
}, indent=2), encoding="utf-8")
print(f"saved -> {output}")
