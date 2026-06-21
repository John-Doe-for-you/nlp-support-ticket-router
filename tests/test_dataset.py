"""Tests for dataset loading, label mapping, and stratified splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from ticket_router.preprocessing.dataset import (
    TARGET_CATEGORIES,
    build_labeled_frame,
    category_distribution,
    map_to_category,
    stratified_split,
)


# ---------------------------------------------------------------------------
# Pure mapping tests (no filesystem)
# ---------------------------------------------------------------------------


def _row(queue: str | None, tag1: str | None, subject: str, body: str = "") -> dict[str, object]:
    return {
        "subject": subject,
        "body": body,
        "queue": queue,
        "tag_1": tag1,
        "tag_2": "",
        "tag_3": "",
        "tag_4": "",
    }


def test_map_to_category_queue_override():
    # Queue is the strongest signal: even if text screams "crash",
    # the Billing queue should win.
    row = _row(
        queue="Billing and Payments",
        tag1=None,
        subject="App crashes on launch",
        body="Critical bug, please fix",
    )
    assert map_to_category(row) == "Billing"


def test_map_to_category_tag1_signal_authentication():
    row = _row(queue=None, tag1="Login", subject="hello", body="")
    assert map_to_category(row) == "Authentication"


def test_map_to_category_tag1_signal_billing():
    row = _row(queue=None, tag1="Billing", subject="hi", body="")
    assert map_to_category(row) == "Billing"


def test_map_to_category_keyword_billing():
    row = _row(
        queue=None,
        tag1=None,
        subject="Double charge on subscription",
        body="I was charged twice for the monthly plan",
    )
    assert map_to_category(row) == "Billing"


def test_map_to_category_keyword_authentication():
    row = _row(
        queue=None,
        tag1=None,
        subject="Can't log in",
        body="I forgot my password and reset link is not coming through",
    )
    assert map_to_category(row) == "Authentication"


def test_map_to_category_keyword_bug_report():
    row = _row(
        queue=None,
        tag1=None,
        subject="App is broken",
        body="The app crashes every time I open it. Stack trace attached.",
    )
    assert map_to_category(row) == "Bug Report"


def test_map_to_category_keyword_feature_request():
    row = _row(
        queue=None,
        tag1=None,
        subject="Feature request",
        body="Could you add dark mode to the dashboard?",
    )
    assert map_to_category(row) == "Feature Request"


def test_map_to_category_keyword_technical_setup():
    row = _row(
        queue=None,
        tag1=None,
        subject="How to install the SDK",
        body="Step by step guide for setting up the API key",
    )
    assert map_to_category(row) == "Technical Setup"


def test_map_to_category_returns_none_when_unmatched():
    row = _row(
        queue="General Inquiry",
        tag1="Feedback",
        subject="Hello team,",
        body="Just wanted to say your product is interesting.",
    )
    assert map_to_category(row) is None


def test_map_to_category_handles_missing_fields_gracefully():
    row = {"subject": "Charge dispute", "body": "refund please", "tag_1": None, "tag_2": None}
    assert map_to_category(row) == "Billing"


# ---------------------------------------------------------------------------
# Stratified split tests
# ---------------------------------------------------------------------------


def _toy_frame() -> pd.DataFrame:
    rng = pd.DataFrame(
        {
            "text": [f"ticket {i}" for i in range(100)],
            "category": (["Billing"] * 20 + ["Authentication"] * 20
                         + ["Bug Report"] * 20 + ["Feature Request"] * 20
                         + ["Technical Setup"] * 20),
        }
    )
    return rng


def test_stratified_split_sizes_are_close_to_70_15_15():
    df = _toy_frame()
    train, val, test = stratified_split(df, seed=42)
    # sklearn's stratified split rounds to whole rows, so allow ±1.
    assert abs(len(train) - 70) <= 1
    assert abs(len(val) - 15) <= 1
    assert abs(len(test) - 15) <= 1


def test_stratified_split_is_deterministic():
    df = _toy_frame()
    t1, v1, te1 = stratified_split(df, seed=42)
    t2, v2, te2 = stratified_split(df, seed=42)
    assert t1.equals(t2)
    assert v1.equals(v2)
    assert te1.equals(te1)


def test_stratified_split_changes_with_seed():
    df = _toy_frame()
    t1, _, _ = stratified_split(df, seed=42)
    t2, _, _ = stratified_split(df, seed=43)
    assert not t1.equals(t2)


def test_stratified_split_preserves_class_proportions():
    df = _toy_frame()
    train, val, test = stratified_split(df, seed=42)
    overall = category_distribution(df)

    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        counts = category_distribution(split)
        # Each target category must appear in every split for this balanced frame.
        for cat in TARGET_CATEGORIES:
            assert counts.get(cat, 0) >= 1, f"{cat} missing from {split_name}"
        # And the per-class share within each split should be roughly 20% (1/5).
        total = sum(counts.values())
        for cat in TARGET_CATEGORIES:
            share = counts.get(cat, 0) / total
            assert 0.15 <= share <= 0.25, f"{cat} share={share:.2f} in {split_name}"
    # Sanity: overall class counts are 20 each.
    assert overall == {c: 20 for c in TARGET_CATEGORIES}


def test_stratified_split_requires_category_column():
    df = pd.DataFrame({"text": ["a", "b", "c"], "label": [1, 2, 3]})
    with pytest.raises(ValueError):
        stratified_split(df)


# ---------------------------------------------------------------------------
# Integration tests against the real raw CSV (skipped if not present)
# ---------------------------------------------------------------------------


def test_build_labeled_frame_returns_only_target_categories(raw_csv_path):
    df = build_labeled_frame(raw_csv_path, english_only=True)
    assert set(df["category"].unique()).issubset(set(TARGET_CATEGORIES))
    assert len(df) > 0
    assert {"text", "category"}.issubset(df.columns)


def test_build_labeled_frame_is_reproducible(raw_csv_path):
    a = build_labeled_frame(raw_csv_path, english_only=True)
    b = build_labeled_frame(raw_csv_path, english_only=True)
    # Same length and same category counts (texts may not be in same order)
    assert a["category"].value_counts().to_dict() == b["category"].value_counts().to_dict()
    assert len(a) == len(b)


def test_stratified_split_on_real_data(raw_csv_path):
    df = build_labeled_frame(raw_csv_path, english_only=True)
    train, val, test = stratified_split(df, seed=42)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    # Every target category should appear in every split (since each has >=1 sample)
    for split in (train, val, test):
        cats = set(split["category"].unique())
        assert cats.issubset(set(TARGET_CATEGORIES))


def test_split_sizes_sum_to_total(raw_csv_path):
    df = build_labeled_frame(raw_csv_path, english_only=True)
    train, val, test = stratified_split(df, seed=42)
    assert len(train) + len(val) + len(test) == len(df)
    # Tolerate 2% deviation from 70/15/15
    total = len(df)
    assert abs(len(train) / total - 0.70) < 0.02
    assert abs(len(val) / total - 0.15) < 0.02
    assert abs(len(test) / total - 0.15) < 0.02