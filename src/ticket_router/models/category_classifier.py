"""TF-IDF + Logistic Regression category classifier.

The classifier is a thin, scikit-learn-native pipeline designed for Day 7:

    cleaner.clean_text  ->  FeatureUnion (word + char TF-IDF)
                         ->  LogisticRegression(class_weight='balanced')

Public API:
    CATEGORIES            : tuple of the 5 locked category names
    build_pipeline(...)   : construct an *untrained* sklearn Pipeline
    CategoryClassifier    : trained wrapper exposing predict / predict_proba /
                            predict_with_confidence; supports save/load via joblib.

Design notes
------------
* Word n-grams (1-2) capture local syntax ("reset password", "double charge").
* Character n-grams (3-5) are robust to typos and morphological variants
  (e.g. "loggin", "billingg"), which is helpful for noisy support tickets.
* `class_weight='balanced'` is the Day 6 locked choice; it counters the
  ~5.5x imbalance between Bug Report (largest) and Technical Setup (smallest).
* LogReg with `liblinear` is fast on sparse text features and supports
  `class_weight` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from ticket_router.preprocessing.cleaner import clean_text

CATEGORIES: tuple[str, ...] = (
    "Billing",
    "Authentication",
    "Bug Report",
    "Feature Request",
    "Technical Setup",
)


def build_pipeline(
    *,
    class_weight: str | dict[str, float] | None = "balanced",
    random_state: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
) -> Pipeline:
    """Return an untrained sklearn Pipeline for category classification.

    The pipeline's `.predict` accepts raw text strings; the `clean_text` step
    is applied as a FunctionTransformer so callers do not have to pre-clean.
    """
    from sklearn.preprocessing import FunctionTransformer

    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=False,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
    )

    features = FeatureUnion(
        [("word", word_tfidf), ("char", char_tfidf)],
        transformer_weights={"word": 1.0, "char": 0.5},
    )

    clf = LogisticRegression(
        solver="liblinear",
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        C=C,
    )

    return Pipeline(
        [
            ("clean", FunctionTransformer(_clean_batch, validate=False)),
            ("features", features),
            ("clf", clf),
        ]
    )


def _clean_batch(texts: Iterable[str]) -> list[str]:
    """Vectorized cleaner used as the first pipeline step."""
    return [clean_text(t) if isinstance(t, str) else "" for t in texts]


@dataclass
class CategoryPrediction:
    """Convenience result for `CategoryClassifier.predict_with_confidence`."""

    category: str
    confidence: float
    scores: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "confidence": float(self.confidence),
            "scores": {k: float(v) for k, v in self.scores.items()},
        }


class CategoryClassifier:
    """Trained wrapper around the category pipeline.

    The wrapper adds a small ergonomic layer on top of the raw sklearn
    Pipeline: a `predict_with_confidence` method that returns a structured
    `CategoryPrediction`, plus `save` / `load` helpers that persist the
    whole pipeline as a single joblib blob.
    """

    pipeline: Pipeline

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @classmethod
    def build(cls, **kwargs: object) -> "CategoryClassifier":
        """Construct a fresh, untrained `CategoryClassifier`."""
        return cls(build_pipeline(**kwargs))

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "CategoryClassifier":
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict(self, texts: Sequence[str] | str) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        return list(self.pipeline.predict(list(texts)))

    def predict_proba(self, texts: Sequence[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline.predict_proba(list(texts))

    def predict_with_confidence(
        self, texts: Sequence[str] | str
    ) -> list[CategoryPrediction]:
        """Predict and return per-class score maps + the top label."""
        if isinstance(texts, str):
            texts = [texts]
        proba = self.predict_proba(texts)
        classes: list[str] = [str(c) for c in self.pipeline.classes_]
        out: list[CategoryPrediction] = []
        for row in proba:
            top_idx = int(np.argmax(row))
            out.append(
                CategoryPrediction(
                    category=classes[top_idx],
                    confidence=float(row[top_idx]),
                    scores={classes[i]: float(row[i]) for i in range(len(classes))},
                )
            )
        return out

    def classes_(self) -> list[str]:
        return [str(c) for c in self.pipeline.classes_]

    def feature_names(self) -> list[str]:
        """Combined word + char feature names in pipeline order."""
        union = self.pipeline.named_steps["features"]
        names: list[str] = []
        for _, step in union.transformer_list:
            names.extend(step.get_feature_names_out())
        return list(names)

    def decision_function(self, texts: Sequence[str] | str) -> np.ndarray:
        """Raw LogReg decision values (length n_samples x n_classes)."""
        if isinstance(texts, str):
            texts = [texts]
        cleaned = _clean_batch(texts)
        features = self.pipeline.named_steps["features"].transform(cleaned)
        return self.pipeline.named_steps["clf"].decision_function(features)

    def save(self, path: str | Path) -> Path:
        """Persist the pipeline to disk; returns the resolved path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "CategoryClassifier":
        return cls(joblib.load(Path(path)))

    def transform(self, texts: Sequence[str]) -> spmatrix:
        """Expose the sparse TF-IDF matrix for downstream inspection/tests."""
        cleaned = _clean_batch(texts)
        return self.pipeline.named_steps["features"].transform(cleaned)


__all__ = [
    "CATEGORIES",
    "build_pipeline",
    "CategoryClassifier",
    "CategoryPrediction",
]
