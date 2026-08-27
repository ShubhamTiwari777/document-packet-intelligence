"""Select a boundary decision rule on validation data, then score it once on held-out test data.

Three rules are compared on identical model probabilities, so the only variable is how scores
become splits:

* `threshold`      -- one global cut-off, swept.
* `expected_count` -- K = round(sum p) splits at the top-K seams (current production). Needs no
                      parameter, but must commit to exactly K, so it drops a confident seam that
                      ranks K+1 and promotes a weak one that ranks K.
* `hybrid`         -- split if p >= ceiling (regardless of rank) OR the seam is in the top K and
                      p >= floor. Both of expected_count's failure directions get an escape hatch.

Leakage discipline. The test manifest is not opened until after selection has finished and the
chosen parameters are printed. Nothing is swept, fitted or compared on test: it is evaluated once,
with frozen parameters, and reported whatever it says. `--split-halves` partitions one manifest
into disjoint tune/test halves by stream id for corpora that ship no separate split.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.stage1_metrics import boundary_lift, grouping_accuracy
from src.features.extractor import packet_feature_rows
from src.features.text_features import TfidfTextEmbedder
from src.openpss_dataset import pages_from_stream
from src.stage1.boundary_classifier import SklearnBoundaryModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/boundary_tabme")
parser.add_argument("--val", required=True)
parser.add_argument("--test", default=None, help="Held-out manifest, opened only after selection.")
parser.add_argument("--split-halves", action="store_true",
                    help="Ignore --test and split --val into disjoint tune/test halves by stream id.")
parser.add_argument("--baseline-threshold", type=float, default=0.6334265856848469)
parser.add_argument("--grid", type=int, default=24, help="Grid points per hybrid parameter.")
parser.add_argument("--bootstrap", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="outputs/benchmarks/decision_rule_tuning.json")
args = parser.parse_args()


# ----------------------------------------------------------------- scoring the packets once
def score_manifest(path: str) -> list[dict]:
    """Model probabilities + labels per packet. Independent of any decision rule."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    model = SklearnBoundaryModel.load(args.model)
    embedder_path = Path(args.model) / "text_embedder.pkl"
    embedder = TfidfTextEmbedder.load(embedder_path) if embedder_path.exists() else None
    metadata = Path(args.model) / "metadata.json"
    synthesize = json.loads(metadata.read_text(encoding="utf-8")).get("synthetic_blocks", False) \
        if metadata.exists() else False
    packets = []
    for stream in manifest["streams"]:
        try:
            pages = pages_from_stream(stream, synthesize)
            rows = packet_feature_rows(pages, stream["stream_id"], embedder)
            if len(rows) != len(stream["boundary_labels"]):
                continue
            packets.append({"stream_id": stream["stream_id"],
                            "probabilities": [float(p) for p in model.predict_proba(rows)],
                            "labels": list(stream["boundary_labels"])})
        except Exception:
            continue
    return packets


# ----------------------------------------------------------------- the three rules
def decide(probabilities: list[float], rule: str, params: dict) -> list[int]:
    if rule == "threshold":
        return [int(p >= params["threshold"]) for p in probabilities]
    count = max(0, min(len(probabilities), round(sum(probabilities))))
    ranked = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:count]
    if rule == "expected_count":
        chosen = set(ranked)
    elif rule == "hybrid":
        chosen = {i for i in ranked if probabilities[i] >= params["floor"]}
        chosen |= {i for i, p in enumerate(probabilities) if p >= params["ceiling"]}
    else:
        raise ValueError(rule)
    return [int(i in chosen) for i in range(len(probabilities))]


def owners(decisions: list[int]) -> list[str]:
    current, out = 0, ["d0"]
    for decision in decisions:
        current += decision
        out.append(f"d{current}")
    return out


def _segments(decisions: list[int]) -> list[int]:
    """Sizes of the contiguous page runs implied by a list of split decisions."""
    sizes, run = [], 1
    for decision in decisions:
        if decision:
            sizes.append(run); run = 1
        else:
            run += 1
    sizes.append(run)
    return sizes


def fast_grouping(decisions: list[int], labels: list[int]) -> float:
    """Pairwise agreement in O(pages) instead of O(pages^2).

    Both segmentations are contiguous runs, so the pair counts follow from segment sizes and
    their overlaps; the reference implementation enumerates every page pair, which is far too
    slow to sweep hundreds of thresholds over 100-page streams. Checked against it below.
    """
    predicted, actual = _segments(decisions), _segments(labels)
    pages = len(decisions) + 1
    total = pages * (pages - 1) // 2
    pairs = lambda size: size * (size - 1) // 2
    together_p = sum(pairs(s) for s in predicted)
    together_a = sum(pairs(s) for s in actual)
    # Overlap sizes of the two run partitions, via a linear merge over their boundaries.
    together_both, i = 0, 0
    start_p = start_a = 0
    pi = ai = 0
    while pi < len(predicted) and ai < len(actual):
        end_p, end_a = start_p + predicted[pi], start_a + actual[ai]
        together_both += pairs(max(0, min(end_p, end_a) - max(start_p, start_a)))
        if end_p <= end_a:
            start_p = end_p; pi += 1
        else:
            start_a = end_a; ai += 1
    agree = total - together_p - together_a + 2 * together_both
    return agree / total if total else 1.0


def evaluate(packets: list[dict], rule: str, params: dict) -> dict:
    predicted, actual, per_packet = [], [], []
    for packet in packets:
        decisions = decide(packet["probabilities"], rule, params)
        predicted += decisions
        actual += packet["labels"]
        per_packet.append(fast_grouping(decisions, packet["labels"]))
    metrics = boundary_lift(predicted, actual)
    metrics["grouping"] = sum(per_packet) / max(len(per_packet), 1)
    metrics["per_packet_grouping"] = per_packet
    metrics["rule"], metrics["params"] = rule, params
    return metrics


def line(label: str, m: dict) -> str:
    return (f"  {label:<46} grouping {m['grouping']:.4f}   F1 {m['f1']:.3f}  "
            f"P {m['precision']:.3f}  R {m['recall']:.3f}  lift {m['lift_over_trivial']:+.3f}")


# ----------------------------------------------------------------- PHASE 1: validation only
rng = random.Random(args.seed)
if args.split_halves:
    everything = score_manifest(args.val)
    everything.sort(key=lambda p: p["stream_id"])
    rng.shuffle(everything)
    half = len(everything) // 2
    val_packets, test_packets = everything[:half], everything[half:]
    print(f"Split one manifest into disjoint halves by stream: "
          f"{len(val_packets)} tune / {len(test_packets)} held-out")
else:
    val_packets = score_manifest(args.val)
    test_packets = None

val_pairs = sum(len(p["labels"]) for p in val_packets)
print(f"\n=== PHASE 1  selection on VALIDATION only ===")
print(f"  {len(val_packets)} packets, {val_pairs} seams, "
      f"{sum(sum(p['labels']) for p in val_packets)/val_pairs:.1%} boundary density")
print(f"  selection metric: page grouping accuracy (pre-committed)\n")

# Verify the fast metric against the reference implementation before trusting any number it
# produces. A wrong-but-fast metric would silently select the wrong rule.
for packet in val_packets[:25]:
    for rule, params in (("expected_count", {}), ("threshold", {"threshold": 0.5})):
        d = decide(packet["probabilities"], rule, params)
        reference = grouping_accuracy(owners(d), owners(packet["labels"]))
        if abs(reference - fast_grouping(d, packet["labels"])) > 1e-9:
            raise SystemExit(f"fast_grouping disagrees with the reference on {packet['stream_id']}")
print("  fast grouping metric verified against the reference implementation on 50 cases")

distinct = sorted({round(p, 6) for packet in val_packets for p in packet["probabilities"]})
if len(distinct) > 240:
    # Sweeping every distinct value is quadratic in corpus size; quantile points cover the same
    # decision surface, since only values between observed scores change any decision.
    step = len(distinct) / 240
    candidates = sorted({distinct[min(len(distinct) - 1, int(i * step))] for i in range(240)})
else:
    candidates = distinct
print(f"  {len(distinct)} distinct probability values, sweeping {len(candidates)} of them\n")

results = {"baseline_threshold": evaluate(val_packets, "threshold", {"threshold": args.baseline_threshold}),
           "expected_count": evaluate(val_packets, "expected_count", {})}
print("  reference points:")
print(line(f"inherited threshold {args.baseline_threshold:.4f} (fitted for the OTHER model)",
           results["baseline_threshold"]))
print(line("expected_count (current production, no parameters)", results["expected_count"]))

best_threshold = None
for value in candidates:
    m = evaluate(val_packets, "threshold", {"threshold": value})
    if best_threshold is None or m["grouping"] > best_threshold["grouping"]:
        best_threshold = m
print("\n  swept:")
print(line(f"best fixed threshold = {best_threshold['params']['threshold']:.4f}", best_threshold))

lo, hi = candidates[0], candidates[-1]
grid = [lo + (hi - lo) * i / (args.grid - 1) for i in range(args.grid)]
best_hybrid = None
for floor in grid:
    for ceiling in grid:
        if ceiling < floor:
            continue
        m = evaluate(val_packets, "hybrid", {"floor": floor, "ceiling": ceiling})
        if best_hybrid is None or m["grouping"] > best_hybrid["grouping"]:
            best_hybrid = m
print(line(f"best hybrid floor={best_hybrid['params']['floor']:.4f} "
           f"ceiling={best_hybrid['params']['ceiling']:.4f}", best_hybrid))

results["tuned_threshold"] = best_threshold
results["tuned_hybrid"] = best_hybrid

winner = max(("tuned_threshold", "tuned_hybrid", "expected_count"),
             key=lambda k: results[k]["grouping"])
locked = {"rule": results[winner]["rule"], "params": results[winner]["params"]}
print(f"\n  >>> SELECTED ON VALIDATION: {winner}  {locked['params']}")
print(f"  >>> parameters are now frozen; the test manifest has not been read yet.")

# ----------------------------------------------------------------- PHASE 2: test, once
test_report = None
if test_packets is None and args.test:
    print(f"\n=== PHASE 2  opening held-out TEST for the first time ===")
    test_packets = score_manifest(args.test)

if test_packets:
    pairs = sum(len(p["labels"]) for p in test_packets)
    print(f"\n=== PHASE 2  single evaluation on HELD-OUT TEST ===")
    print(f"  {len(test_packets)} packets, {pairs} seams, "
          f"{sum(sum(p['labels']) for p in test_packets)/pairs:.1%} boundary density\n")
    test_report = {}
    for name, rule, params in (
            ("expected_count (production)", "expected_count", {}),
            (f"tuned threshold {best_threshold['params']['threshold']:.4f}", "threshold", best_threshold["params"]),
            ("tuned hybrid", "hybrid", best_hybrid["params"]),
            (f"inherited threshold {args.baseline_threshold:.4f}", "threshold", {"threshold": args.baseline_threshold})):
        m = evaluate(test_packets, rule, params)
        test_report[name] = m
        marker = "  <-- selected" if (rule == locked["rule"] and params == locked["params"]) else ""
        print(line(name, m) + marker)

    # Bootstrap over packets: is the selected rule's gain over production real?
    production = test_report["expected_count (production)"]["per_packet_grouping"]
    chosen = test_report[[k for k in test_report if locked["rule"] in k or
                          (locked["rule"] == "expected_count" and "production" in k)][0]]["per_packet_grouping"]
    deltas = []
    n = len(production)
    for _ in range(args.bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(chosen[i] - production[i] for i in idx) / n)
    deltas.sort()
    low, high = deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas))]
    point = sum(c - p for c, p in zip(chosen, production)) / n
    print(f"\n  selected rule minus production, page grouping:")
    print(f"    {point:+.4f}   95% bootstrap CI [{low:+.4f}, {high:+.4f}]  "
          f"({'significant' if low > 0 or high < 0 else 'NOT significant - includes zero'})")

for value in results.values():
    value.pop("per_packet_grouping", None)
if test_report:
    for value in test_report.values():
        value.pop("per_packet_grouping", None)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"model": args.model, "val": args.val, "test": args.test,
                              "selection_metric": "page_grouping_accuracy",
                              "locked": locked, "validation": results, "test_results": test_report},
                             indent=2), encoding="utf-8")
print(f"\n-> {output}")
