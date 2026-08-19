"""Compare gradient-boosting implementations for boundary detection on identical features.

The shipped learner (scikit-learn HistGradientBoosting) was inherited rather than chosen, and
`xgboost` sat in requirements.txt without ever being imported. This script settles the question by
measurement: same cached features, same isotonic calibration, same packet-level split, same
expected-count decision rule -- only the learner changes.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.stage1_metrics import boundary_lift, grouping_accuracy
from src.features.extractor import FEATURE_NAMES, load_feature_cache
from src.stage1.grouping import decide_boundaries

parser = argparse.ArgumentParser()
parser.add_argument("--features", default="data/processed/cache/openpss_flow/boundary_features.json")
parser.add_argument("--manifest", default="data/raw/openpss/train/manifest.json")
parser.add_argument("--output", default="outputs/benchmarks/boundary_learners.json")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

rows = load_feature_cache(args.features)
manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

# Rebuild labels by aligning cached rows with the manifest, grouped by packet.
labels_by_packet = {stream["stream_id"]: stream["boundary_labels"] for stream in manifest["streams"]}
packets: dict[str, list[dict]] = {}
for row in rows:
    packets.setdefault(row["packet_id"], []).append(row)

usable = [(pid, rs, labels_by_packet[pid]) for pid, rs in packets.items()
          if pid in labels_by_packet and len(rs) == len(labels_by_packet[pid])]
usable.sort(key=lambda item: item[0])
split = int(len(usable) * 0.8)
train, test = usable[:split], usable[split:]
print(f"packets: {len(usable)} ({len(train)} train / {len(test)} test)")


def matrix(group):
    return ([[r[n] for n in FEATURE_NAMES] for _, rs, _ in group for r in rs],
            [l for _, _, ls in group for l in ls])


X_train, y_train = matrix(train)
X_test, y_test = matrix(test)
print(f"pairs: {len(X_train)} train / {len(X_test)} test  |  positives {sum(y_train)}/{sum(y_test)}")


def build_estimators():
    from sklearn.ensemble import HistGradientBoostingClassifier
    estimators = {"sklearn_hist_gbm": HistGradientBoostingClassifier(
        random_state=args.seed, max_iter=150, learning_rate=0.08, max_leaf_nodes=15)}
    try:
        from xgboost import XGBClassifier
        estimators["xgboost"] = XGBClassifier(
            random_state=args.seed, n_estimators=150, learning_rate=0.08, max_leaves=15,
            tree_method="hist", eval_metric="logloss", verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        estimators["lightgbm"] = LGBMClassifier(
            random_state=args.seed, n_estimators=150, learning_rate=0.08, num_leaves=15, verbose=-1)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        estimators["catboost"] = CatBoostClassifier(
            random_seed=args.seed, iterations=150, learning_rate=0.08, depth=6, verbose=False)
    except ImportError:
        pass
    return estimators


def evaluate(model) -> dict:
    probabilities = [float(p[1]) for p in model.predict_proba(X_test)]
    offset = 0
    accuracies = []
    for _, rs, ls in test:
        window = probabilities[offset:offset + len(rs)]
        offset += len(rs)
        decisions = decide_boundaries(window, 0.5, "expected_count")
        predicted, actual, p, a = [], [], 0, 0
        for index in range(len(rs) + 1):
            if index > 0:
                p += decisions[index - 1]
                a += ls[index - 1]
            predicted.append(f"p{p}")
            actual.append(f"t{a}")
        accuracies.append(grouping_accuracy(predicted, actual))
    metrics = boundary_lift(decide_boundaries(probabilities, 0.5, "expected_count"), y_test)
    metrics["grouping_accuracy"] = round(sum(accuracies) / max(len(accuracies), 1), 4)
    return metrics


results = []
for name, estimator in build_estimators().items():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.utils.class_weight import compute_sample_weight
    weights = compute_sample_weight("balanced", y_train)
    calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv=3)
    start = time.perf_counter()
    try:
        calibrated.fit(X_train, y_train, sample_weight=weights)
    except Exception as exc:
        results.append({"learner": name, "status": f"failed: {type(exc).__name__}: {str(exc)[:120]}"})
        print(f"  {name:<18} FAILED: {str(exc)[:80]}")
        continue
    train_seconds = time.perf_counter() - start
    buffer = io.BytesIO(); pickle.dump(calibrated, buffer)
    metrics = evaluate(calibrated)
    results.append({"learner": name, "train_seconds": round(train_seconds, 2),
                    "model_bytes": buffer.tell(), **metrics, "status": "measured"})
    print(f"  {name:<18} F1 {metrics['f1']:.4f}  lift {metrics['lift_over_trivial']:+.4f}  "
          f"grouping {metrics['grouping_accuracy']:.4f}  train {train_seconds:6.1f}s  "
          f"{buffer.tell()/1e6:5.1f} MB")

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(
    {"features": len(FEATURE_NAMES), "train_pairs": len(X_train), "test_pairs": len(X_test),
     "results": results}, indent=2), encoding="utf-8")
print(f"\nWrote {args.output}")
