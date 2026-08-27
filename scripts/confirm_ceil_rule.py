"""Single, final confirmation of the locked ceil rule on the TABME++ test split.

PRE-REGISTRATION. The candidate was chosen on TABME++ validation in scripts/simulate_soft_k.py
and is frozen here before the test manifest is opened:

    Rule 2 candidate  :  K = ceil(sum of probabilities), split at the top K seams
    Comparators       :  K = round(sum of probabilities)   -- the historical rule
                         p >= 0.185011                     -- the threshold already locked and
                                                              shipped, selected on the same
                                                              validation split
    Primary metric    :  page grouping accuracy
    Validation result :  ceil 0.9637  |  round 0.9472  |  threshold 0.9603

Nothing below is swept, fitted or selected. The three rules are fully specified above; this file
evaluates them once and reports whatever comes out.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.stage1_metrics import boundary_lift
from src.features.extractor import packet_feature_rows
from src.features.text_features import TfidfTextEmbedder
from src.openpss_dataset import pages_from_stream
from src.stage1.boundary_classifier import SklearnBoundaryModel

MODEL = "models/boundary_tabme"
TEST_MANIFEST = "data/raw/tabme_test/manifest.json"
LOCKED_THRESHOLD = 0.185011
CONFIDENT = 0.5

parser = argparse.ArgumentParser()
parser.add_argument("--bootstrap", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="outputs/benchmarks/ceil_rule_confirmation.json")
args = parser.parse_args()

print(__doc__.strip())
print("\n" + "=" * 88)
print("LOCKED before opening test:  K = ceil(sum p)   [primary metric: page grouping accuracy]")
print("=" * 88)


def _ranked(p):
    return sorted(range(len(p)), key=lambda i: p[i], reverse=True)


def rule_round(p):
    k = max(0, min(len(p), round(sum(p))))
    return set(_ranked(p)[:k])


def rule_ceil(p):
    k = max(0, min(len(p), math.ceil(sum(p) - 1e-9)))
    return set(_ranked(p)[:k])


def rule_threshold(p):
    return {i for i, v in enumerate(p) if v >= LOCKED_THRESHOLD}


RULES = [("round(sum p)  [historical]", rule_round),
         ("ceil(sum p)   [locked candidate]", rule_ceil),
         (f"p >= {LOCKED_THRESHOLD}  [in production]", rule_threshold)]


def segments(d):
    sizes, run = [], 1
    for x in d:
        if x: sizes.append(run); run = 1
        else: run += 1
    sizes.append(run)
    return sizes


def grouping(d, y):
    a, b = segments(d), segments(y)
    n = len(d) + 1
    total = n * (n - 1) // 2
    pair = lambda s: s * (s - 1) // 2
    both = sa = sb = i = j = 0
    while i < len(a) and j < len(b):
        ea, eb = sa + a[i], sb + b[j]
        both += pair(max(0, min(ea, eb) - max(sa, sb)))
        if ea <= eb: sa = ea; i += 1
        else: sb = eb; j += 1
    return (total - sum(map(pair, a)) - sum(map(pair, b)) + 2 * both) / total if total else 1.0


# ------------------------------------------------------------------ open test, once
manifest = json.loads(Path(TEST_MANIFEST).read_text(encoding="utf-8"))
model = SklearnBoundaryModel.load(MODEL)
embedder = TfidfTextEmbedder.load(Path(MODEL) / "text_embedder.pkl")
synthesize = json.loads((Path(MODEL) / "metadata.json").read_text(encoding="utf-8")).get("synthetic_blocks", False)

packets = []
for stream in manifest["streams"]:
    try:
        pages = pages_from_stream(stream, synthesize)
        rows = packet_feature_rows(pages, stream["stream_id"], embedder)
        if len(rows) != len(stream["boundary_labels"]):
            continue
        packets.append({"p": [float(v) for v in model.predict_proba(rows)],
                        "y": list(stream["boundary_labels"])})
    except Exception:
        continue

pairs = sum(len(x["y"]) for x in packets)
print(f"\nTABME++ TEST: {len(packets)} packets, {pairs} seams, "
      f"{sum(sum(x['y']) for x in packets)/pairs:.1%} boundary density\n")


def evaluate(fn):
    pred, act, per_packet = [], [], []
    starved = forced = 0
    for packet in packets:
        p, y = packet["p"], packet["y"]
        chosen = fn(p)
        d = [int(i in chosen) for i in range(len(p))]
        starved += sum(1 for i in range(len(p)) if y[i] == 1 and not d[i] and p[i] >= CONFIDENT)
        forced += sum(1 for i in range(len(p)) if d[i] and p[i] < CONFIDENT)
        pred += d; act += y; per_packet.append(grouping(d, y))
    m = boundary_lift(pred, act)
    m["grouping"] = sum(per_packet) / len(per_packet)
    m["starved"], m["forced"], m["splits"] = starved, forced, sum(pred)
    m["per_packet"] = per_packet
    return m


results = {name: evaluate(fn) for name, fn in RULES}

print(f"  {'rule':<36}{'grouping':>10}{'P':>8}{'R':>8}{'F1':>8}{'lift':>9}{'splits':>8}{'starved':>9}{'forced':>8}")
for name, m in results.items():
    print(f"  {name:<36}{m['grouping']:>10.4f}{m['precision']:>8.3f}{m['recall']:>8.3f}"
          f"{m['f1']:>8.3f}{m['lift_over_trivial']:>+9.3f}{m['splits']:>8}{m['starved']:>9}{m['forced']:>8}")
print(f"\n  trivial always-split baseline: F1 {results[RULES[0][0]]['trivial_f1']:.3f} "
      f"at {results[RULES[0][0]]['base_rate']:.1%} density")

# ------------------------------------------------------------------ bootstrap over packets
rng = random.Random(args.seed)
n = len(packets)
names = [name for name, _ in RULES]
comparisons = [(names[1], names[0]), (names[1], names[2])]
print(f"\n  paired bootstrap over packets ({args.bootstrap} resamples), page grouping accuracy:")
intervals = {}
for a, b in comparisons:
    left, right = results[a]["per_packet"], results[b]["per_packet"]
    point = sum(x - y for x, y in zip(left, right)) / n
    deltas = []
    for _ in range(args.bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(left[i] - right[i] for i in idx) / n)
    deltas.sort()
    lo, hi = deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas))]
    verdict = "significant" if lo > 0 or hi < 0 else "NOT significant (includes zero)"
    intervals[f"{a} minus {b}"] = {"point": round(point, 5), "ci_low": round(lo, 5),
                                   "ci_high": round(hi, 5), "significant": lo > 0 or hi < 0}
    print(f"    {a:<36} minus {b}")
    print(f"      {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   {verdict}")

for m in results.values():
    m.pop("per_packet", None)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(
    {"preregistered_candidate": "ceil(sum p)", "primary_metric": "page_grouping_accuracy",
     "validation": {"ceil": 0.9637, "round": 0.9472, "threshold": 0.9603},
     "test_manifest": TEST_MANIFEST, "packets": n, "seams": pairs,
     "results": results, "bootstrap": intervals}, indent=2), encoding="utf-8")
print(f"\n-> {args.output}")
