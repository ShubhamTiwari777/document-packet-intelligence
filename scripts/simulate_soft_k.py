"""Simulate soft-K variants of expected_count on validation data only.

expected_count treats K = round(sum p) as a hard budget: exactly the top K seams are cut. The
diagnostic in scripts/analyse_expected_count.py showed the count estimate itself is good where the
model is calibrated (English: Pearson r 0.90, K exact 76%), and that the damage comes from
*enforcing* it -- 47% of missed English boundaries were seams the model scored >= 0.5 but could not
afford. The question here is whether K can be kept as evidence while ceasing to be a constraint.

Every variant below is derived rather than tuned. Under a calibrated model the expected change in
correct decisions from cutting a seam is p - (1 - p) = 2p - 1, which is positive exactly when
p > 0.5; that is the Bayes rule for symmetric loss, not a fitted parameter. The same 0.5 already
sits inside expected_count via round(). No variant introduces a constant that was not already
implied by the existing rule.

The TABME test manifest is deliberately not referenced anywhere in this file. Dutch corpora appear
only as a regression check on the opposite failure mode, never as a selection signal.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.stage1_metrics import boundary_lift
from src.features.extractor import packet_feature_rows
from src.features.text_features import TfidfTextEmbedder
from src.openpss_dataset import pages_from_stream
from src.stage1.boundary_classifier import SklearnBoundaryModel

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs/benchmarks/soft_k_simulation.json")
args = parser.parse_args()

# SELECTION set (English) first; the rest are regression checks only.
SELECTION = ("TABME++ validation", "English", "models/boundary_tabme", "data/raw/tabme_val/manifest.json")
CHECKS = [
    ("OpenPSS test_full", "Dutch", "models/boundary_shortpackets", "data/raw/openpss/test_full/manifest.json"),
    ("DocSplit our200", "Dutch", "models/boundary_shortpackets", "data/raw/docsplit_benchmark/manifest.json"),
]


def score(model_dir: str, manifest_path: str) -> list[dict]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    model = SklearnBoundaryModel.load(model_dir)
    embedder_path = Path(model_dir) / "text_embedder.pkl"
    embedder = TfidfTextEmbedder.load(embedder_path) if embedder_path.exists() else None
    metadata = Path(model_dir) / "metadata.json"
    synthesize = json.loads(metadata.read_text(encoding="utf-8")).get("synthetic_blocks", False) \
        if metadata.exists() else False
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
    return packets


# ----------------------------------------------------------------- the variants
def _ranked(p): return sorted(range(len(p)), key=lambda i: p[i], reverse=True)


def v_baseline(p):
    k = max(0, min(len(p), round(sum(p))))
    return set(_ranked(p)[:k])


def v_ceil(p):
    """The measured calibration bias on English is negative, so round() systematically
    under-counts. ceil() is the smallest change that stops truncating downward."""
    k = max(0, min(len(p), math.ceil(sum(p) - 1e-9)))
    return set(_ranked(p)[:k])


def v_add_bayes(p):
    """K as a floor: keep the top K, then admit any rejected seam the model already calls
    more-likely-than-not. Pass two of the two-pass rule."""
    return v_baseline(p) | {i for i, v in enumerate(p) if v > 0.5}


def v_drop_weak(p):
    """K as a ceiling only: never cut a seam the model calls less-likely-than-not."""
    return {i for i in v_baseline(p) if p[i] > 0.5}


def v_soft_both(p):
    """K purely advisory in both directions."""
    return {i for i in v_baseline(p) if p[i] > 0.5} | {i for i, v in enumerate(p) if v > 0.5}


def v_bayes_only(p):
    """No K at all -- the reference that shows whether K contributes anything."""
    return {i for i, v in enumerate(p) if v > 0.5}


def v_residual(p):
    """Two-pass top-up driven by the rule's own arithmetic.

    After taking the top K, the expected number of boundaries still sitting in the rejected set is
    the sum of their probabilities. If that expectation rounds to at least one more boundary, take
    the strongest remaining seam and re-evaluate. This reuses round()'s own 0.5 convention and
    introduces nothing new.
    """
    order = _ranked(p)
    k = max(0, min(len(p), round(sum(p))))
    chosen, cursor = set(order[:k]), k
    while cursor < len(order):
        if sum(p[i] for i in order[cursor:]) >= 0.5:
            chosen.add(order[cursor]); cursor += 1
        else:
            break
    return chosen


def v_threshold_185(p):
    return {i for i, v in enumerate(p) if v >= 0.185011}


VARIANTS = [
    ("expected_count (baseline)", v_baseline),
    ("A  ceil(sum p) instead of round", v_ceil),
    ("B  top-K + admit p>0.5  (two-pass)", v_add_bayes),
    ("C  top-K - drop p<0.5", v_drop_weak),
    ("D  soft-K both directions", v_soft_both),
    ("E  residual-mass top-up", v_residual),
    ("F  p>0.5 only, no K", v_bayes_only),
    ("*  tuned threshold 0.185 (reference)", v_threshold_185),
]


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


def run(packets, fn):
    pred, act, groups = [], [], []
    starved = forced = 0
    for packet in packets:
        p, y = packet["p"], packet["y"]
        chosen = fn(p)
        d = [int(i in chosen) for i in range(len(p))]
        starved += sum(1 for i in range(len(p)) if y[i] == 1 and not d[i] and p[i] >= 0.5)
        forced += sum(1 for i in range(len(p)) if d[i] and p[i] < 0.5)
        pred += d; act += y; groups.append(grouping(d, y))
    m = boundary_lift(pred, act)
    m["grouping"] = sum(groups) / max(len(groups), 1)
    m["starved"], m["forced"] = starved, forced
    m["splits"] = sum(pred)
    return m


def report(label, language, model_dir, manifest, selection: bool) -> dict:
    packets = score(model_dir, manifest)
    tag = "SELECTION SET" if selection else "regression check only"
    print(f"\n{'='*94}\n{label}  ({language}, {len(packets)} packets)   [{tag}]\n{'='*94}")
    print(f"  {'variant':<38}{'grouping':>9}{'P':>7}{'R':>7}{'F1':>7}{'lift':>8}"
          f"{'splits':>8}{'starved':>9}{'forced':>8}")
    out = {}
    base = None
    for name, fn in VARIANTS:
        m = run(packets, fn)
        out[name] = {k: m[k] for k in ("precision", "recall", "f1", "lift_over_trivial",
                                       "grouping", "starved", "forced", "splits")}
        if base is None:
            base = m["grouping"]
        delta = m["grouping"] - base
        print(f"  {name:<38}{m['grouping']:>9.4f}{m['precision']:>7.3f}{m['recall']:>7.3f}"
              f"{m['f1']:>7.3f}{m['lift_over_trivial']:>+8.3f}{m['splits']:>8}"
              f"{m['starved']:>9}{m['forced']:>8}"
              + ("" if abs(delta) < 1e-9 else f"   {delta:+.4f}"))
    return out


results = {SELECTION[0]: report(*SELECTION, selection=True)}
for check in CHECKS:
    if Path(check[3]).exists():
        results[check[0]] = report(*check, selection=False)

# ----------------------------------------------------------------- the concrete starved case
print(f"\n{'='*94}\nThe hand-authored stress packet: does each variant recover the 0.4627 seam?\n{'='*94}")
from src.config import load_config
from src.ingestion.pdf_parser import PDFParser
config = load_config("config/default.yaml")
pages = PDFParser(config.ingestion, config.runtime).parse(
    "data/samples/stress_packet.pdf", Path(r"C:\Users\shiva\AppData\Local\Temp\claude\cmp\softk"))
embedder = TfidfTextEmbedder.load("models/boundary_tabme/text_embedder.pkl")
rows = packet_feature_rows(pages, "stress", embedder)
p = [float(v) for v in SklearnBoundaryModel.load("models/boundary_tabme").predict_proba(rows)]
truth = json.loads(Path("data/samples/stress_packet_ground_truth.json").read_text(encoding="utf-8"))["documents"]
starts = {min(d["pages"]) for d in truth}
y = [1 if i + 2 in starts else 0 for i in range(len(p))]
print(f"  sum(p) = {sum(p):.4f} -> K = {round(sum(p))};  true boundaries = {sum(y)}")
print(f"  {'variant':<38}{'docs':>6}{'grouping':>10}{'recall':>8}   seams cut")
for name, fn in VARIANTS:
    chosen = fn(p)
    d = [int(i in chosen) for i in range(len(p))]
    rec = sum(1 for i in range(len(p)) if d[i] and y[i]) / max(sum(y), 1)
    cut = ",".join(f"{i+1}|{i+2}" for i in range(len(p)) if d[i])
    print(f"  {name:<38}{sum(d)+1:>6}{grouping(d, y):>10.4f}{rec:>8.3f}   {cut}")

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\n-> {args.output}")
