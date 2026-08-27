"""Interrogate the assumption behind the expected_count decision rule.

expected_count sets K = round(sum of per-seam probabilities) and splits at the K highest-scoring
seams. That rests on one statistical claim: for calibrated probabilities, the sum over a packet
estimates how many boundaries the packet contains. This script tests that claim directly and
separates the two ways the rule can then fail:

  starved  -- a true boundary the model scored confidently is rejected because it ranked K+1.
  forced   -- a seam the model scored weakly is selected because K demanded another boundary.

Nothing here is fitted or selected. The confidence level used to label a seam "confident" or
"weak" is fixed at 0.5 in advance, and the comparison threshold rule uses the 0.185011 already
chosen on validation in scripts/tune_decision_rule.py. Diagnostics are reported on validation as
well as test so that any follow-up change can be motivated without reading the test split.
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

CONFIDENT = 0.5          # pre-specified, not fitted
TUNED_THRESHOLD = 0.185011  # selected on TABME validation, frozen

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="outputs/benchmarks/expected_count_analysis.json")
parser.add_argument("--bins", type=int, default=10)
args = parser.parse_args()

CORPORA = [
    ("TABME++ validation", "English", "models/boundary_tabme", "data/raw/tabme_val/manifest.json"),
    ("TABME++ test", "English", "models/boundary_tabme", "data/raw/tabme_test/manifest.json"),
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
            packets.append({"id": stream["stream_id"],
                            "p": [float(v) for v in model.predict_proba(rows)],
                            "y": list(stream["boundary_labels"])})
        except Exception:
            continue
    return packets


def segments(decisions: list[int]) -> list[int]:
    sizes, run = [], 1
    for d in decisions:
        if d:
            sizes.append(run); run = 1
        else:
            run += 1
    sizes.append(run)
    return sizes


def grouping(decisions: list[int], labels: list[int]) -> float:
    a, b = segments(decisions), segments(labels)
    n = len(decisions) + 1
    total = n * (n - 1) // 2
    pair = lambda s: s * (s - 1) // 2
    both, sa, sb, i, j = 0, 0, 0, 0, 0
    while i < len(a) and j < len(b):
        ea, eb = sa + a[i], sb + b[j]
        both += pair(max(0, min(ea, eb) - max(sa, sb)))
        if ea <= eb: sa = ea; i += 1
        else: sb = eb; j += 1
    return (total - sum(map(pair, a)) - sum(map(pair, b)) + 2 * both) / total if total else 1.0


def expected_count(p: list[float]) -> tuple[list[int], int]:
    k = max(0, min(len(p), round(sum(p))))
    ranked = sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:k]
    chosen = set(ranked)
    return [int(i in chosen) for i in range(len(p))], k


def analyse(label: str, language: str, model_dir: str, manifest_path: str) -> dict:
    packets = score(model_dir, manifest_path)
    rows, pred_all, act_all, groupings = [], [], [], []
    starved = starved_confident = forced = forced_wrong = 0
    k_exact = k_under = k_over = 0
    k_abs_error = 0

    for packet in packets:
        p, y = packet["p"], packet["y"]
        decisions, k = expected_count(p)
        selected = [i for i, d in enumerate(decisions) if d]
        rejected = [i for i, d in enumerate(decisions) if not d]
        true_count = sum(y)

        misses = [i for i in rejected if y[i] == 1]
        false_positives = [i for i in selected if y[i] == 0]
        starved += len(misses)
        starved_confident += sum(1 for i in misses if p[i] >= CONFIDENT)
        weak = [i for i in selected if p[i] < CONFIDENT]
        forced += len(weak)
        forced_wrong += sum(1 for i in weak if y[i] == 0)

        error = k - true_count
        k_abs_error += abs(error)
        k_exact += error == 0
        k_under += error < 0
        k_over += error > 0

        rows.append({
            "packet": packet["id"], "seams": len(p),
            "actual_boundaries": true_count,
            "sum_probabilities": round(sum(p), 4),
            "K": k, "K_error": error,
            "correct": len(selected) - len(false_positives),
            "missed": len(misses), "false": len(false_positives),
            "lowest_selected": round(min((p[i] for i in selected), default=float("nan")), 4),
            "highest_rejected": round(max((p[i] for i in rejected), default=float("nan")), 4),
        })
        pred_all += decisions; act_all += y
        groupings.append(grouping(decisions, y))

    metrics = boundary_lift(pred_all, act_all)
    metrics["grouping"] = round(sum(groupings) / max(len(groupings), 1), 4)

    # threshold rule, for comparison only -- parameter frozen on validation
    t_pred = [int(v >= TUNED_THRESHOLD) for packet in packets for v in packet["p"]]
    t_metrics = boundary_lift(t_pred, act_all)
    t_metrics["grouping"] = round(sum(grouping([int(v >= TUNED_THRESHOLD) for v in packet["p"]], packet["y"])
                                      for packet in packets) / max(len(packets), 1), 4)

    # --- does sum(p) estimate the boundary count? ---
    sums = [r["sum_probabilities"] for r in rows]
    truths = [r["actual_boundaries"] for r in rows]
    n = len(rows)
    mean_s, mean_t = sum(sums) / n, sum(truths) / n
    cov = sum((s - mean_s) * (t - mean_t) for s, t in zip(sums, truths))
    var_s = sum((s - mean_s) ** 2 for s in sums)
    var_t = sum((t - mean_t) ** 2 for t in truths)
    pearson = cov / math.sqrt(var_s * var_t) if var_s and var_t else 0.0

    # --- probability calibration ---
    flat = [(v, label_) for packet in packets for v, label_ in zip(packet["p"], packet["y"])]
    bins, ece, brier = [], 0.0, sum((v - label_) ** 2 for v, label_ in flat) / len(flat)
    for b in range(args.bins):
        lo, hi = b / args.bins, (b + 1) / args.bins
        bucket = [(v, l) for v, l in flat if (v >= lo and v < hi) or (b == args.bins - 1 and v == 1.0)]
        if not bucket:
            bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": 0}); continue
        conf = sum(v for v, _ in bucket) / len(bucket)
        freq = sum(l for _, l in bucket) / len(bucket)
        ece += len(bucket) / len(flat) * abs(conf - freq)
        bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": len(bucket),
                     "mean_predicted": round(conf, 4), "observed_rate": round(freq, 4),
                     "gap": round(freq - conf, 4)})

    return {
        "label": label, "language": language, "model": model_dir, "packets": n,
        "expected_count": {k: metrics[k] for k in
                           ("precision", "recall", "f1", "base_rate", "trivial_f1",
                            "lift_over_trivial", "grouping")},
        "tuned_threshold_0.185": {k: t_metrics[k] for k in
                                  ("precision", "recall", "f1", "lift_over_trivial", "grouping")},
        "count_estimation": {
            "mean_sum_p": round(mean_s, 3), "mean_true": round(mean_t, 3),
            "global_bias": round(mean_s - mean_t, 3),
            "pearson_r": round(pearson, 4),
            "mean_abs_K_error": round(k_abs_error / n, 3),
            "K_exact_pct": round(100 * k_exact / n, 1),
            "K_under_pct": round(100 * k_under / n, 1),
            "K_over_pct": round(100 * k_over / n, 1),
        },
        "calibration": {"ECE": round(ece, 4), "brier": round(brier, 4), "bins": bins},
        "failure_attribution": {
            "total_missed": starved,
            "missed_while_confident": starved_confident,
            "missed_while_confident_pct_of_misses": round(100 * starved_confident / max(starved, 1), 1),
            "forced_weak_selections": forced,
            "forced_weak_that_were_wrong": forced_wrong,
            "forced_weak_precision": round(1 - forced_wrong / max(forced, 1), 4),
        },
        "packets_detail": rows,
    }


reports = []
for label, language, model_dir, manifest_path in CORPORA:
    if not Path(manifest_path).exists():
        print(f"skipping {label}: {manifest_path} not present"); continue
    report = analyse(label, language, model_dir, manifest_path)
    reports.append(report)

    e, t, c, cal, f = (report["expected_count"], report["tuned_threshold_0.185"],
                       report["count_estimation"], report["calibration"], report["failure_attribution"])
    print(f"\n{'='*78}\n{label}  ({language}, {report['packets']} packets, {model_dir})\n{'='*78}")
    print(f"  expected_count      P {e['precision']:.3f}  R {e['recall']:.3f}  F1 {e['f1']:.3f}  "
          f"grouping {e['grouping']:.4f}  lift {e['lift_over_trivial']:+.3f}")
    print(f"  threshold 0.185     P {t['precision']:.3f}  R {t['recall']:.3f}  F1 {t['f1']:.3f}  "
          f"grouping {t['grouping']:.4f}  lift {t['lift_over_trivial']:+.3f}")
    print(f"\n  -- does sum(p) estimate the boundary count? --")
    print(f"     mean sum(p) {c['mean_sum_p']:.3f}   mean true {c['mean_true']:.3f}   "
          f"bias {c['global_bias']:+.3f}   Pearson r {c['pearson_r']:.4f}")
    print(f"     K exact {c['K_exact_pct']}%   under {c['K_under_pct']}%   over {c['K_over_pct']}%   "
          f"mean |K-true| {c['mean_abs_K_error']:.3f}")
    print(f"\n  -- probability calibration --   ECE {cal['ECE']:.4f}   Brier {cal['brier']:.4f}")
    print(f"     {'bin':<10}{'n':>7}{'predicted':>12}{'observed':>11}{'gap':>9}")
    for b in cal["bins"]:
        if not b["count"]:
            continue
        print(f"     {b['bin']:<10}{b['count']:>7}{b['mean_predicted']:>12.3f}"
              f"{b['observed_rate']:>11.3f}{b['gap']:>+9.3f}")
    print(f"\n  -- where the forced top-K selection goes wrong --")
    print(f"     boundaries missed entirely        {f['total_missed']}")
    print(f"       of which the model scored >=0.5 {f['missed_while_confident']}  "
          f"({f['missed_while_confident_pct_of_misses']}% of misses)  <- starved by K")
    print(f"     weak seams (<0.5) forced in       {f['forced_weak_selections']}")
    print(f"       of which were wrong             {f['forced_weak_that_were_wrong']}  "
          f"(precision {f['forced_weak_precision']:.3f})")

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"confident_level": CONFIDENT, "tuned_threshold": TUNED_THRESHOLD,
                              "reports": reports}, indent=2), encoding="utf-8")
print(f"\n-> {output}  (per-packet detail included)")
