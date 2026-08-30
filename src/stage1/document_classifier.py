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
    def build_vectorizer(cls, word_features: int = 20_000, char_features: int = 30_000):
        """Word n-grams unioned with character n-grams.

        The training corpus is OCR of scanned documents, so a word-level vocabulary is brittle in
        exactly the way that matters: `Invoice` read as `lnvoice` and `Total` as `TotaI` are simply
        out of vocabulary, and the evidence they carried is lost. Character n-grams degrade instead
        of failing -- most of the substrings survive a one-character misread -- so the two views
        cover different failure modes and are used together.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.pipeline import FeatureUnion  # type: ignore
        return FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                     max_features=word_features, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     max_features=char_features, sublinear_tf=True)),
        ])

    @classmethod
    def train(cls, texts: list[str], labels: list[str], seed: int = 42) -> "TfidfDocumentClassifier":
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.pipeline import Pipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for the TF-IDF classifier.") from exc
        if len(texts) != len(labels) or len(set(labels)) < 2: raise ValueError("Training needs aligned text and at least two classes.")
        # 20k word features measured marginally *better* than 200k (acc 0.807 vs 0.803, macro-F1
        # 0.786 vs 0.775) at a tenth of the on-disk size -- the larger space mostly added noisy OCR
        # tokens. The character view is added on top rather than instead: see build_vectorizer.
        pipeline = Pipeline([("features", cls.build_vectorizer()),
                             ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))])
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


class LayoutTextDocumentClassifier:
    """Document type from what the page says *and* what it looks like.

    Text alone is a weak signal on scanned documents: the corpus is OCR output, and the classes
    that fail worst are the ones distinguished by shape rather than vocabulary -- a presentation, a
    form and a scientific report share most of their words but almost none of their geometry.
    The layout half is 21 geometry-only descriptors (see features/document_shape.py), scaled and
    concatenated onto the sparse text matrix.

    Layout is required rather than optional at inference. Passing `None` and quietly substituting
    zeros would reproduce the exact failure this project already paid for once, where a model was
    trained on features that meant something different in production.
    """

    def __init__(self, vectorizer: Any, scaler: Any, classifier: Any):
        self.vectorizer, self.scaler, self.classifier = vectorizer, scaler, classifier

    @classmethod
    def train(cls, texts: list[str], layouts: list[list[float]], labels: list[str],
              seed: int = 42) -> "LayoutTextDocumentClassifier":
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for the layout classifier.") from exc
        if not (len(texts) == len(layouts) == len(labels)):
            raise ValueError("Texts, layout vectors and labels must be aligned.")
        if len(set(labels)) < 2:
            raise ValueError("Training needs at least two classes.")
        vectorizer = TfidfDocumentClassifier.build_vectorizer()
        text_matrix = vectorizer.fit_transform(texts)
        scaler = StandardScaler()
        layout_matrix = scaler.fit_transform(layouts)
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        classifier.fit(cls._combine(text_matrix, layout_matrix), labels)
        return cls(vectorizer, scaler, classifier)

    @staticmethod
    def _combine(text_matrix: Any, layout_matrix: Any) -> Any:
        from scipy.sparse import csr_matrix, hstack  # type: ignore
        return hstack([text_matrix, csr_matrix(layout_matrix)]).tocsr()

    @property
    def labels(self) -> list[str]:
        return [str(label) for label in self.classifier.classes_]

    def _matrix(self, texts: list[str], layouts: list[list[float]] | None) -> Any:
        if layouts is None:
            raise ValueError(
                "This model was trained with layout features; predicting without them would score "
                "documents on a feature space the model never saw. Pass layout vectors from "
                "features.document_shape.document_shape_features.")
        if len(texts) != len(layouts):
            raise ValueError("Texts and layout vectors must be aligned.")
        return self._combine(self.vectorizer.transform(texts), self.scaler.transform(layouts))

    def predict_distribution(self, texts: list[str], layouts: list[list[float]] | None = None) -> list[dict[str, float]]:
        classes = self.labels
        rows = self.classifier.predict_proba(self._matrix(texts, layouts))
        return [dict(zip(classes, (float(value) for value in row))) for row in rows]

    def predict(self, texts: list[str], layouts: list[list[float]] | None = None) -> list[ClassPrediction]:
        results: list[ClassPrediction] = []
        for distribution in self.predict_distribution(texts, layouts):
            label, probability = max(distribution.items(), key=lambda item: item[1])
            results.append(ClassPrediction(label, float(probability), "trained_layout",
                                           _top_alternatives(distribution)))
        return results

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as handle:
            pickle.dump({"kind": "layout_text", "vectorizer": self.vectorizer,
                         "scaler": self.scaler, "classifier": self.classifier}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "LayoutTextDocumentClassifier":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        return cls(payload["vectorizer"], payload["scaler"], payload["classifier"])


class HybridDocumentClassifier:
    """Trained taxonomy + lexicon extension, with explicit abstention below `min_confidence`."""

    def __init__(self, trained: TfidfDocumentClassifier | None, lexicon: LexiconDocumentClassifier | None = None, min_confidence: float = 0.35, extension_min_confidence: float | None = None):
        self.trained = trained
        self.lexicon = lexicon or LexiconDocumentClassifier()
        self.min_confidence = min_confidence
        # Deliberately stricter than `min_confidence`: clearing this bar lets an out-of-taxonomy
        # type override the trained model outright, so it should require solid evidence.
        self.extension_min_confidence = extension_min_confidence if extension_min_confidence is not None else max(0.5, min_confidence)

    def predict(self, texts: list[str], layouts: list[list[float]] | None = None) -> list[ClassPrediction]:
        """`layouts` is forwarded only to a model that was trained with it.

        Text-only models take no layout argument, so passing one is harmless; a layout model
        raises rather than scoring against zeros if the caller omits it.
        """
        lexicon_results = self.lexicon.predict(texts)
        if self.trained is None:
            trained_results: list[Any] = [None] * len(texts)
        elif isinstance(self.trained, LayoutTextDocumentClassifier):
            trained_results = self.trained.predict(texts, layouts)
        else:
            trained_results = self.trained.predict(texts)
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
    # A layout model is a dict payload rather than a bare pipeline; sniff rather than trusting the
    # declared kind, so a config that names the wrong one cannot silently load a mismatched model.
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and payload.get("kind") == "layout_text":
        trained: Any = LayoutTextDocumentClassifier(payload["vectorizer"], payload["scaler"], payload["classifier"])
        active, needs_layout = "hybrid_layout_text_lexicon", True
    else:
        trained = TfidfDocumentClassifier(payload)
        active, needs_layout = "hybrid_tfidf_lexicon", False
    status.update({"active": active, "trained_model_loaded": True, "model_path": str(path),
                   "requires_layout": needs_layout,
                   "trained_labels": trained.labels, "extension_labels": sorted(EXTENSION_CLASSES)})
    return HybridDocumentClassifier(trained, min_confidence=min_confidence), status
