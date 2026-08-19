"""Document-type classification.

Three components, composed by `build_document_classifier`:

* `TfidfDocumentClassifier`      -- trained TF-IDF + logistic regression over the RVL-CDIP
                                    document-type taxonomy. Emits real calibrated probabilities.
* `LexiconDocumentClassifier`    -- weighted-phrase prior for document types that have no public
                                    labeled training data (passport, bank statement). Emits a
                                    normalized distribution gated by absolute evidence, NOT the
                                    old "fraction of this class's keyword list" score, which was
                                    not comparable across classes and could not express ambiguity.
* `HybridDocumentClassifier`     -- reconciles the two and applies `classification.min_confidence`
                                    abstention, so a weakly-supported guess surfaces as `unknown`
                                    instead of a confident-looking wrong label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math
import pickle

from src.features.text_features import cosine, hashed_embedding


# Weighted evidence phrases. Weights encode specificity: "issuing authority" is near-conclusive
# for a passport, whereas a bare "total" appears in almost any financial page.
LEXICON: dict[str, dict[str, float]] = {
    "invoice": {
        "tax invoice": 3.0, "invoice number": 3.0, "invoice no": 2.5, "bill to": 2.5,
        "amount due": 2.5, "payment terms": 1.5, "invoice": 2.0, "subtotal": 1.5, "total": 0.5,
    },
    "resume": {
        "curriculum vitae": 3.0, "professional summary": 2.5, "work experience": 2.5,
        "employment history": 2.5, "technical skills": 2.0, "certifications": 1.5,
        "education": 1.0, "skills": 1.0, "experience": 0.5,
    },
    "passport": {
        "issuing authority": 3.0, "travel document": 3.0, "passport no": 3.0, "passport": 2.5,
        "place of birth": 2.0, "given names": 2.0, "nationality": 2.0, "date of birth": 1.5,
        "surname": 1.5,
    },
    "bank_statement": {
        "bank statement": 3.0, "account statement": 3.0, "opening balance": 3.0,
        "closing balance": 3.0, "statement period": 2.5, "total debits": 2.5,
        "total credits": 2.5, "account holder": 2.0, "account number": 2.0, "transaction": 1.0,
    },
}

# Types the trained taxonomy (RVL-CDIP) cannot represent at all. For a document of one of these
# types the trained model must assign its probability mass to some unrelated in-taxonomy class,
# so a confident lexicon hit is strictly better evidence than a confident trained prediction.
EXTENSION_CLASSES = {"passport", "bank_statement"}

# Evidence needed before the lexicon is allowed to sound confident. Chosen so that a single
# generic phrase cannot dominate, while two-to-three specific phrases can.
EVIDENCE_SATURATION = 4.0


@dataclass
class ClassPrediction:
    label: str
    confidence: float
    source: str = "unknown"
    alternatives: list[dict[str, Any]] = field(default_factory=list)


def _top_alternatives(distribution: dict[str, float], limit: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [{"label": label, "probability": round(float(probability), 4)} for label, probability in ranked]


class LexiconDocumentClassifier:
    """Weighted-phrase prior producing a comparable, evidence-gated probability."""

    def __init__(self, lexicon: dict[str, dict[str, float]] | None = None, saturation: float = EVIDENCE_SATURATION):
        self.lexicon = lexicon or LEXICON
        self.saturation = saturation

    def score(self, text: str) -> tuple[dict[str, float], float]:
        """Return (normalized distribution, absolute evidence of the winning class)."""
        lowered = text.lower()
        evidence = {
            label: sum(weight for phrase, weight in phrases.items() if phrase in lowered)
            for label, phrases in self.lexicon.items()
        }
        total = sum(evidence.values())
        if total <= 0:
            return {label: 0.0 for label in self.lexicon}, 0.0
        distribution = {label: value / total for label, value in evidence.items()}
        return distribution, max(evidence.values())

    def predict(self, texts: list[str]) -> list[ClassPrediction]:
        results: list[ClassPrediction] = []
        for text in texts:
            distribution, evidence = self.score(text)
            if evidence <= 0:
                results.append(ClassPrediction("unknown", 0.0, "lexicon_no_evidence", []))
                continue
            label = max(distribution.items(), key=lambda item: item[1])[0]
            # Dominance over rival classes, damped by how much absolute evidence was actually seen.
            saturation = evidence / (evidence + self.saturation)
            results.append(ClassPrediction(
                label=label,
                confidence=float(distribution[label] * saturation),
                source="lexicon",
                alternatives=_top_alternatives(distribution),
            ))
        return results


# Backwards-compatible alias: the historical name used across scripts and configs.
KeywordDocumentClassifier = LexiconDocumentClassifier


class TfidfDocumentClassifier:
    def __init__(self, pipeline: Any): self.pipeline = pipeline

    @classmethod
    def train(cls, texts: list[str], labels: list[str], seed: int = 42) -> "TfidfDocumentClassifier":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.pipeline import Pipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for the TF-IDF classifier.") from exc
        if len(texts) != len(labels) or len(set(labels)) < 2: raise ValueError("Training needs aligned text and at least two classes.")
        # 20k features measured marginally *better* than 200k (acc 0.807 vs 0.803, macro-F1 0.786
        # vs 0.775) at a tenth of the on-disk size -- the larger space mostly added noisy OCR
        # tokens. Keeping the small model per the resource-efficiency criterion.
        pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True)), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))])
        pipeline.fit(texts, labels)
        return cls(pipeline)

    @property
    def labels(self) -> list[str]:
        return [str(label) for label in self.pipeline.classes_]

    def predict_distribution(self, texts: list[str]) -> list[dict[str, float]]:
        classes = self.labels
        return [dict(zip(classes, (float(value) for value in row))) for row in self.pipeline.predict_proba(texts)]

    def predict(self, texts: list[str]) -> list[ClassPrediction]:
        results: list[ClassPrediction] = []
        for distribution in self.predict_distribution(texts):
            label, probability = max(distribution.items(), key=lambda item: item[1])
            results.append(ClassPrediction(label, float(probability), "trained", _top_alternatives(distribution)))
        return results

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as handle: pickle.dump(self.pipeline, handle)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfDocumentClassifier":
        with Path(path).open("rb") as handle: return cls(pickle.load(handle))


class HybridDocumentClassifier:
    """Trained taxonomy + lexicon extension, with explicit abstention below `min_confidence`."""

    def __init__(self, trained: TfidfDocumentClassifier | None, lexicon: LexiconDocumentClassifier | None = None, min_confidence: float = 0.35, extension_min_confidence: float | None = None):
        self.trained = trained
        self.lexicon = lexicon or LexiconDocumentClassifier()
        self.min_confidence = min_confidence
        # Deliberately stricter than `min_confidence`: clearing this bar lets an out-of-taxonomy
        # type override the trained model outright, so it should require solid evidence.
        self.extension_min_confidence = extension_min_confidence if extension_min_confidence is not None else max(0.5, min_confidence)

    def predict(self, texts: list[str]) -> list[ClassPrediction]:
        lexicon_results = self.lexicon.predict(texts)
        trained_results = self.trained.predict(texts) if self.trained else [None] * len(texts)
        results: list[ClassPrediction] = []
        for lexical, trained in zip(lexicon_results, trained_results):
            results.append(self._reconcile(lexical, trained))
        return results

    def _reconcile(self, lexical: ClassPrediction, trained: ClassPrediction | None) -> ClassPrediction:
        if trained is None:
            if lexical.confidence >= self.min_confidence:
                return ClassPrediction(lexical.label, lexical.confidence, "lexicon_only", lexical.alternatives)
            return ClassPrediction("unknown", lexical.confidence, "abstained_no_trained_model", lexical.alternatives)
        # An out-of-taxonomy type can only ever come from the lexicon. The trained model's
        # confidence is NOT comparable here -- it is a distribution over a taxonomy that excludes
        # this class, so it is confidently wrong by construction. Given solid lexicon evidence,
        # the extension label therefore wins outright rather than being compared numerically.
        if lexical.label in EXTENSION_CLASSES and lexical.confidence >= self.extension_min_confidence:
            return ClassPrediction(lexical.label, lexical.confidence, "lexicon_extension", lexical.alternatives)
        if trained.confidence >= self.min_confidence:
            return ClassPrediction(trained.label, trained.confidence, "trained", trained.alternatives)
        if lexical.confidence >= self.min_confidence:
            return ClassPrediction(lexical.label, lexical.confidence, "lexicon_backoff", lexical.alternatives)
        return ClassPrediction("unknown", max(trained.confidence, lexical.confidence), "abstained_low_confidence", trained.alternatives)


class EmbeddingCentroidDocumentClassifier:
    """A lightweight embedding-based classifier baseline, independent of TF-IDF."""

    def __init__(self, centroids: dict[str, list[float]]): self.centroids = centroids

    @classmethod
    def train(cls, texts: list[str], labels: list[str]) -> "EmbeddingCentroidDocumentClassifier":
        if len(texts) != len(labels) or not texts: raise ValueError("Training texts and labels must be non-empty and aligned.")
        buckets: dict[str, list[list[float]]] = {}
        for text, label in zip(texts, labels): buckets.setdefault(label, []).append(hashed_embedding(text))
        return cls({label: [sum(vector[i] for vector in vectors) / len(vectors) for i in range(len(vectors[0]))] for label, vectors in buckets.items()})

    def predict(self, texts: list[str]) -> list[ClassPrediction]:
        results: list[ClassPrediction] = []
        for text in texts:
            scores = {label: cosine(hashed_embedding(text), vector) for label, vector in self.centroids.items()}
            label, similarity = max(scores.items(), key=lambda item: item[1])
            results.append(ClassPrediction(label, max(0.0, min(1.0, (similarity + 1) / 2)), "embedding_centroid", _top_alternatives(scores)))
        return results

    def save(self, path: str | Path) -> None:
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"kind": "embedding_centroid", "centroids": self.centroids}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingCentroidDocumentClassifier":
        import json
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["centroids"])


def build_document_classifier(model_path: str | Path | None, min_confidence: float = 0.35, declared_kind: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Resolve the configured classifier and report honestly what was actually loaded.

    The previous inline selection in the pipeline fell back to keyword matching *silently*
    whenever `classification.model_path` was unset, even though the config declared
    `tfidf_logistic_regression` -- so a misconfigured run looked identical to a trained one.
    This returns a status describing what was requested versus what is in use.
    """
    status: dict[str, Any] = {"declared_kind": declared_kind, "min_confidence": min_confidence}
    if not model_path:
        status.update({"active": "lexicon_only", "trained_model_loaded": False,
                       "warning": f"classification.model_path is not set; '{declared_kind}' was declared but only the lexicon prior is active."})
        return HybridDocumentClassifier(None, min_confidence=min_confidence), status
    path = Path(model_path)
    if not path.exists():
        status.update({"active": "lexicon_only", "trained_model_loaded": False,
                       "warning": f"classification.model_path '{path}' does not exist; falling back to the lexicon prior."})
        return HybridDocumentClassifier(None, min_confidence=min_confidence), status
    if str(path).lower().endswith(".json"):
        status.update({"active": "embedding_centroid", "trained_model_loaded": True, "model_path": str(path)})
        return EmbeddingCentroidDocumentClassifier.load(path), status
    trained = TfidfDocumentClassifier.load(path)
    status.update({"active": "hybrid_tfidf_lexicon", "trained_model_loaded": True, "model_path": str(path),
                   "trained_labels": trained.labels, "extension_labels": sorted(EXTENSION_CLASSES)})
    return HybridDocumentClassifier(trained, min_confidence=min_confidence), status
