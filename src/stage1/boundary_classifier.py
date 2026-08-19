"""Trainable, lightweight boundary models plus transparent baselines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import pickle

from src.features.extractor import FEATURE_NAMES


@dataclass
class BoundaryPrediction:
    probability: float
    predicted_boundary: bool


class WeightedFusionBoundary:
    """Validation-tunable transparent baseline, with no ML dependency."""
    feature_names = FEATURE_NAMES

    def __init__(self, text_weight: float = 0.55, visual_weight: float = 0.15, heuristic_weight: float = 0.30):
        self.text_weight, self.visual_weight, self.heuristic_weight = text_weight, visual_weight, heuristic_weight

    def predict_proba(self, rows: list[dict[str, float]]) -> list[float]:
        probabilities: list[float] = []
        for row in rows:
            heuristic = (row["page_number_reset"] + (1 - row["page_number_continuation"]) + (1 - row["header_similarity"]) + (1 - row["footer_similarity"])) / 4
            score = self.text_weight * row["text_delta"] + self.visual_weight * row["visual_delta"] + self.heuristic_weight * heuristic
            probabilities.append(max(0.0, min(1.0, score)))
        return probabilities

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"kind": "weighted_fusion", "weights": self.__dict__, "feature_names": FEATURE_NAMES}, indent=2), encoding="utf-8")


class SklearnBoundaryModel:
    """Our trained model wrapper. Uses HistGradientBoosting for a reliable CPU default."""

    def __init__(self, estimator: Any, feature_names: list[str] | None = None):
        self.estimator = estimator
        # A model must score with the feature list it was TRAINED on, not whatever the current
        # code defines. Without this, adding a feature to FEATURE_NAMES silently breaks every
        # previously saved model — the estimator receives a wider matrix than it was fitted with.
        self.feature_names = feature_names or FEATURE_NAMES

    @classmethod
    def train(cls, rows: list[dict[str, float]], labels: list[int], seed: int = 42, calibrate: bool = False, class_weight: str | None = "balanced") -> "SklearnBoundaryModel":
        if len(rows) != len(labels) or not rows:
            raise ValueError("Non-empty aligned training rows and labels are required.")
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
            from sklearn.utils.class_weight import compute_sample_weight  # type: ignore
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required to train the boundary classifier.") from exc
        if len(set(labels)) < 2:
            raise ValueError("Boundary training labels require both boundary and non-boundary examples.")
        matrix = [[row[name] for name in FEATURE_NAMES] for row in rows]  # training defines the set
        # "balanced" reweights the training set to an effective 50/50 prior. That is a poor match
        # when boundaries are ~11% of adjacent pairs in deployment: it biases the model toward
        # predicting boundaries and depresses precision. Passing None keeps the natural prior.
        sample_weight = compute_sample_weight(class_weight, labels) if class_weight else None
        estimator = HistGradientBoostingClassifier(random_state=seed, max_iter=150, learning_rate=0.08, max_leaf_nodes=15)
        if calibrate:
            # Raw HistGradientBoosting probabilities on an imbalanced boundary/non-boundary split are
            # sharply skewed toward 0 even for true positives (see boundary_validation reports), which
            # makes any fixed decision threshold unreliable. Isotonic calibration on held-out folds fixes
            # the probability scale without changing the ranking the base estimator already learned.
            from sklearn.calibration import CalibratedClassifierCV  # type: ignore
            calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv=3)
            calibrated.fit(matrix, labels, sample_weight=sample_weight)
            return cls(calibrated)
        estimator.fit(matrix, labels, sample_weight=sample_weight)
        return cls(estimator)

    def predict_proba(self, rows: list[dict[str, float]]) -> list[float]:
        matrix = [[row[name] for name in self.feature_names] for row in rows]
        learned = [float(item[1]) for item in self.estimator.predict_proba(matrix)]
        # A printed page number resetting (e.g. back to "1") is a near-unambiguous, domain-
        # invariant boundary cue -- unlike the learned score, it doesn't depend on the visual/
        # textual style of whatever corpus the classifier happened to be trained on. Let it
        # override a muted learned probability instead of being outvoted by it.
        return [max(probability, 0.85) if row.get("page_number_reset", 0.0) >= 1.0 else probability for probability, row in zip(learned, rows)]

    def save(self, directory: str | Path, metadata: dict[str, Any]) -> None:
        output = Path(directory); output.mkdir(parents=True, exist_ok=True)
        with (output / "boundary_model.pkl").open("wb") as handle:
            pickle.dump(self.estimator, handle)
        (output / "feature_names.json").write_text(json.dumps(self.feature_names, indent=2), encoding="utf-8")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "SklearnBoundaryModel":
        target = Path(directory)
        names_path = target / "feature_names.json"
        names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else None
        with (target / "boundary_model.pkl").open("rb") as handle:
            return cls(pickle.load(handle), names)


def predict(model: Any, rows: list[dict[str, float]], threshold: float) -> list[BoundaryPrediction]:
    return [BoundaryPrediction(probability=probability, predicted_boundary=probability >= threshold) for probability in model.predict_proba(rows)]