"""Tests for the TF-IDF + LogReg category classifier (Day 7).

Coverage:

* Structural: `build_pipeline` returns a sklearn Pipeline with the expected
  named steps and Hyperparameters, and the wrapper exposes the right methods.
* Cleaning integration: the pipeline lowercases/strips/collapses whitespace
  before vectorizing (we verify via the cleaner transformer's output).
* Semantic smoke tests: a handful of unambiguous, hand-written tickets per
  category must be predicted correctly by a *real* trained model.
* Toy training round-trip: a tiny in-memory training set is enough to teach
  the wrapper to recover the right label on a held-out example of the same
  class. This guards against wiring regressions without requiring the full
  corpus.
* save / load round-trip via joblib.
* Probability sanity: predicted label == argmax of predict_proba; confidence
  in [0, 1]; score map covers every class exactly once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import FeatureUnion, Pipeline

from ticket_router.models.category_classifier import (
    CATEGORIES,
    CategoryClassifier,
    build_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_train_corpus() -> dict[str, list[str]]:
    """Tiny per-category corpus used to teach the classifier on a fresh fit."""
    return {
        "Billing": [
            "I was charged twice for my subscription this month.",
            "Please refund my last invoice, the payment was duplicated.",
            "My credit card on file was declined for the renewal.",
            "I need an itemized receipt for the plan I paid for.",
            "Why was I overcharged on the subscription fee?",
        ],
        "Authentication": [
            "I can't log in to my account, password reset isn't working.",
            "Two factor authentication code is not arriving on my phone.",
            "My account is locked after too many failed login attempts.",
            "I forgot my password and the reset email never came through.",
            "Single sign on with our SSO provider keeps failing.",
        ],
        "Bug Report": [
            "The app crashes every time I open the settings page.",
            "Getting an internal server error 500 when uploading a file.",
            "After the latest update the dashboard is blank.",
            "The export feature is broken, it freezes mid-download.",
            "Data loss: my saved items disappeared after the sync.",
        ],
        "Feature Request": [
            "Could you add dark mode to the dashboard please.",
            "It would be great if the app supported bulk import from CSV.",
            "Please add an integration with Slack for notifications.",
            "I wish there was a way to schedule reports automatically.",
            "Would love a roadmap view of upcoming features.",
        ],
        "Technical Setup": [
            "How do I install the SDK on a fresh Ubuntu machine?",
            "Step by step guide for configuring the API key in my .env.",
            "Where do I configure the docker container for production?",
            "Tutorial for setting up the kubernetes deployment please.",
            "How can I configure the CI/CD pipeline with GitHub Actions?",
        ],
    }


@pytest.fixture(scope="module")
def trained_on_small(small_train_corpus: dict[str, list[str]]) -> CategoryClassifier:
    clf = CategoryClassifier.build()
    texts: list[str] = []
    labels: list[str] = []
    for cat, docs in small_train_corpus.items():
        texts.extend(docs)
        labels.extend([cat] * len(docs))
    return clf.fit(texts, labels)


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_categories_locked_to_five() -> None:
    assert CATEGORIES == (
        "Billing",
        "Authentication",
        "Bug Report",
        "Feature Request",
        "Technical Setup",
    )


def _feature_names(features: FeatureUnion) -> list[str]:
    return [name for name, _ in features.transformer_list]


def test_build_pipeline_has_expected_steps() -> None:
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    step_names = list(pipe.named_steps.keys())
    assert step_names == ["clean", "features", "clf"]

    features = pipe.named_steps["features"]
    assert isinstance(features, FeatureUnion)
    assert _feature_names(features) == ["word", "char"]

    word_vec = features.transformer_list[0][1]
    char_vec = features.transformer_list[1][1]
    assert word_vec.analyzer == "word"
    assert word_vec.ngram_range == (1, 2)
    assert char_vec.analyzer == "char_wb"
    assert char_vec.ngram_range == (3, 5)

    clf = pipe.named_steps["clf"]
    assert clf.solver == "liblinear"
    assert clf.class_weight == "balanced"
    assert clf.random_state == 42
    assert clf.max_iter == 1000


def test_build_factory_returns_unfitted_pipeline() -> None:
    pipe = build_pipeline()
    word_vec = pipe.named_steps["features"].transformer_list[0][1]
    char_vec = pipe.named_steps["features"].transformer_list[1][1]
    # Vectorizers only gain `vocabulary_` after fit
    assert not hasattr(word_vec, "vocabulary_")
    assert not hasattr(char_vec, "vocabulary_")
    assert not hasattr(pipe.named_steps["clf"], "classes_")


# ---------------------------------------------------------------------------
# Cleaner integration: pipeline lowercases before vectorizing
# ---------------------------------------------------------------------------


def test_pipeline_applies_cleaner_first(trained_on_small: CategoryClassifier) -> None:
    """A raw uppercase ticket must still get the right prediction."""
    text = "I WAS CHARGED TWICE FOR MY SUBSCRIPTION."
    pred = trained_on_small.predict(text)
    assert pred == ["Billing"]


def test_pipeline_strips_html_and_urls(trained_on_small: CategoryClassifier) -> None:
    text = (
        "<p>I can't <b>log in</b> to my account.</p> "
        "See https://example.com/help for more info."
    )
    pred = trained_on_small.predict(text)
    assert pred == ["Authentication"]


def test_pipeline_handles_empty_string() -> None:
    """Empty input should not crash; the wrapper returns a valid class."""
    clf = CategoryClassifier.build()
    # Fit on enough docs that min_df=2 retains some vocabulary; then check that
    # an empty string still produces a valid (if arbitrary) prediction.
    texts = [
        "paid invoice twice today",
        "refund my last payment now",
        "reset my password please",
        "my login is broken again",
        "the app crashes on launch",
        "the dashboard is broken too",
        "could you add dark mode please",
        "please add a new export feature",
        "how do i install the sdk",
        "how to configure the api key",
    ]
    labels = ["Billing", "Billing", "Authentication", "Authentication",
              "Bug Report", "Bug Report", "Feature Request", "Feature Request",
              "Technical Setup", "Technical Setup"]
    clf.fit(texts, labels)
    out = clf.predict("")
    assert len(out) == 1
    assert out[0] in CATEGORIES


def test_pipeline_handles_non_string_input_gracefully() -> None:
    """Non-string items should be coerced to empty strings, not crash."""
    clf = CategoryClassifier.build()
    texts = [
        "paid invoice twice today",
        "refund my last payment now",
        "reset my password please",
        "my login is broken again",
        "the app crashes on launch",
        "the dashboard is broken too",
        "could you add dark mode please",
        "please add a new export feature",
        "how do i install the sdk",
        "how to configure the api key",
    ]
    labels = ["Billing", "Billing", "Authentication", "Authentication",
              "Bug Report", "Bug Report", "Feature Request", "Feature Request",
              "Technical Setup", "Technical Setup"]
    clf.fit(texts, labels)
    out = clf.predict([None, 123, ""])  # type: ignore[list-item]
    assert len(out) == 3
    assert all(p in CATEGORIES for p in out)


# ---------------------------------------------------------------------------
# Semantic smoke tests per category (one canonical ticket each)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I was charged twice for my subscription.", "Billing"),
        ("My login is broken, password reset does nothing.", "Authentication"),
        ("The app keeps crashing on launch.", "Bug Report"),
        ("Could you please add dark mode?", "Feature Request"),
        ("How do I install the SDK on Ubuntu?", "Technical Setup"),
    ],
)
def test_canonical_ticket_classified_correctly(
    trained_on_small: CategoryClassifier, text: str, expected: str
) -> None:
    pred = trained_on_small.predict(text)
    assert pred == [expected]


# ---------------------------------------------------------------------------
# Toy training round-trip & API checks
# ---------------------------------------------------------------------------


def test_predict_proba_shape_and_argmax(trained_on_small: CategoryClassifier) -> None:
    texts = ["charged twice", "can't log in", "app crashes", "add dark mode", "install sdk"]
    proba = trained_on_small.predict_proba(texts)
    assert proba.shape == (5, 5)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-6)

    preds = trained_on_small.predict(texts)
    for i, row in enumerate(proba):
        assert preds[i] == trained_on_small.classes_()[int(np.argmax(row))]


def test_predict_with_confidence_structure(trained_on_small: CategoryClassifier) -> None:
    out = trained_on_small.predict_with_confidence(["charged twice on my card"])
    assert len(out) == 1
    pred = out[0]
    assert pred.category in CATEGORIES
    assert 0.0 <= pred.confidence <= 1.0
    assert set(pred.scores.keys()) == set(CATEGORIES)
    assert all(0.0 <= v <= 1.0 for v in pred.scores.values())
    # Confidence must equal the top score in `scores`
    assert pred.confidence == pytest.approx(max(pred.scores.values()))


def test_classes_method_matches_constants(trained_on_small: CategoryClassifier) -> None:
    classes = trained_on_small.classes_()
    assert set(classes) == set(CATEGORIES)
    assert len(classes) == len(CATEGORIES)


def test_feature_names_contains_word_and_char(
    trained_on_small: CategoryClassifier,
) -> None:
    names = trained_on_small.feature_names()
    assert len(names) > 100
    # char_wb features look like 'log gin' (with spaces); word ones are lowercase.
    assert any(" " in n for n in names) or any(n.startswith("__") or " " in n for n in names)


def test_transform_returns_sparse(trained_on_small: CategoryClassifier) -> None:
    from scipy.sparse import issparse

    X = trained_on_small.transform(["charged twice on my card"])
    assert issparse(X)
    assert X.shape[0] == 1
    assert X.shape[1] > 0


# ---------------------------------------------------------------------------
# save / load round-trip via joblib
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(
    trained_on_small: CategoryClassifier, tmp_path: Path
) -> None:
    target = tmp_path / "model.joblib"
    trained_on_small.save(target)
    assert target.exists()

    reloaded = CategoryClassifier.load(target)
    assert reloaded.predict(["charged twice for my subscription"]) == ["Billing"]
    assert reloaded.classes_() == trained_on_small.classes_()


def test_save_creates_parent_dirs(
    trained_on_small: CategoryClassifier, tmp_path: Path
) -> None:
    nested = tmp_path / "a" / "b" / "c" / "model.joblib"
    trained_on_small.save(nested)
    assert nested.exists()


# ---------------------------------------------------------------------------
# Optional: train on the real processed split (skip if not built yet)
# ---------------------------------------------------------------------------


def test_train_on_processed_split_meets_targets(processed_dir: Path) -> None:
    """Fit on the full train split, assert val accuracy >= 0.88 and macro-F1 >= 0.80.

    Skips if the processed CSVs are not present (they are gitignored).
    """
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    if not (train_csv.exists() and val_csv.exists()):
        pytest.skip("Processed splits not present; run prepare_data.py first")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    clf = CategoryClassifier.build().fit(
        train_df["text"].astype(str).tolist(),
        train_df["category"].astype(str).tolist(),
    )
    val_pred = clf.predict(val_df["text"].astype(str).tolist())
    val_true = val_df["category"].astype(str).tolist()

    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(val_true, val_pred)
    macro = f1_score(val_true, val_pred, average="macro", zero_division=0)
    assert acc >= 0.88, f"val accuracy {acc:.4f} below target 0.88"
    assert macro >= 0.80, f"val macro F1 {macro:.4f} below target 0.80"
