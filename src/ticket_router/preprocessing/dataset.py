"""Dataset loading, label mapping, and stratified splitting.

Maps the raw HuggingFace `Tobi-Bueck/customer-support-tickets` schema to the
five locked target categories and produces deterministic 70/15/15 stratified
splits.

Public API:
    TARGET_CATEGORIES      : tuple of the 5 locked category names
    map_to_category(row)   : dict-like row -> str category or None if dropped
    load_raw(path)         : pandas DataFrame of the raw CSV
    build_labeled_frame(path) : DataFrame with columns [text, category] (None dropped)
    stratified_split(df, seed=42) : (train_df, val_df, test_df) at 70/15/15

The mapping is rule-based and uses three signals in order of precedence:
    1. queue  (the dataset's pre-assigned support team)
    2. tag_1  (free-text label like "Billing", "Account", "Security")
    3. keyword scan of subject + body

Rows we cannot confidently place in one of the five target categories are
dropped. We keep only English rows for quality and reproducibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_CATEGORIES: Final[tuple[str, ...]] = (
    "Billing",
    "Authentication",
    "Bug Report",
    "Feature Request",
    "Technical Setup",
)

_RAW_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "subject",
    "body",
    "queue",
    "type",
    "tag_1",
    "language",
)


# ---------------------------------------------------------------------------
# Keyword tables. Lowercased; matched on subject + body + tag columns.
# Order of checks inside map_to_category matters: more specific first.
# ---------------------------------------------------------------------------

_AUTH_KEYWORDS: Final[tuple[str, ...]] = (
    "login", "log in", "log-in", "logon", "sign in", "sign-in", "signin",
    "password", "passcode", "pin code",
    "two-factor", "2fa", "two factor", "two-factor authentication",
    "authentication", "auth code", "verification code", "otp",
    "reset my password", "reset password", "forgot password", "forgot my password",
    "can't access", "cannot access", "unable to access", "unable to log in",
    "unable to login", "cant log in", "cant login",
    "account locked", "account is locked", "account suspended",
    "session expired", "session timeout", "token expired", "access token",
    "single sign-on", "sso", "okta", "active directory",
    "mfa", "multi-factor",
    "wrong password", "incorrect password", "invalid credentials",
    "lost my password", "recover account", "account recovery",
)

_BUG_KEYWORDS: Final[tuple[str, ...]] = (
    "bug", "defect", "broken", "crash", "crashes", "crashed", "crashing",
    "freezes", "freezing", "hangs", "hanging", "hangup",
    "error message", "error code", "exception", "stack trace",
    "not working", "doesn't work", "does not work", "stopped working",
    "is broken", "got broken",
    "blank screen", "white screen", "black screen", "blue screen",
    "data loss", "data lost", "lost data",
    "corrupted", "corruption", "corrupt file",
    "500 error", "404", "http 500", "http 404", "http 503",
    "outage", "downtime", "service is down", "system is down",
    "can't open", "cannot open", "unable to open", "won't open",
    "keeps crashing", "keeps closing",
    "unresponsive", "not responding",
    "glitch", "glitching",
    "regression", "regressed",
)

_FEATURE_KEYWORDS: Final[tuple[str, ...]] = (
    "feature request", "feature suggestion", "suggestion:",
    "could you add", "can you add", "would you add", "please add",
    "would be great if", "would love if", "wish there was",
    "wishlist", "roadmap", "enhancement",
    "support for", "add support for", "support the following",
    "integrate with", "integration with", "integrate ",
    "ability to", "option to",
    "is it possible to", "are there plans to",
    "dark mode", "light mode",
    "export to", "import from",
    "bulk ", "bulk-",
)

_SETUP_KEYWORDS: Final[tuple[str, ...]] = (
    "how to", "how do i", "how can i", "how do you",
    "install", "installation", "installing",
    "setup", "set up", "set-up",
    "configure", "configuration", "configuring",
    "step by step", "instructions for",
    "tutorial", "guide", "walkthrough",
    "where do i", "where can i", "where should i",
    "onboarding", "getting started",
    "sdk", "api key", "api keys",
    "environment variable", "env var", ".env",
    "docker", "kubernetes", "k8s",
    "ci/cd", "pipeline",
    "integration guide",
)

_BILLING_KEYWORDS: Final[tuple[str, ...]] = (
    "invoice", "invoicing", "billing", "bill",
    "charged", "charge", "charges", "double charge", "charged twice",
    "refund", "refunds", "refunded",
    "payment", "payments", "pay",
    "subscription", "subscribe", "renewal", "renew", "auto-renew",
    "pricing", "price", "cost", "quote",
    "credit card", "debit card", "payment method", "payment methods",
    "tax", "vat", "gst",
    "discount", "coupon", "promo code", "promo",
    "receipt", "receipts",
    "plan change", "upgrade plan", "downgrade plan", "change plan",
    "trial", "free trial",
    "money", "usd", "eur", "$", "€", "£",
    "billing cycle", "billing period",
    "overcharge", "overcharged", "undercharged",
)


# Mapping from raw `queue` values to candidate target category. These are the
# high-confidence mappings used before any keyword scan. Anything not in this
# table is treated as ambiguous and falls through to keyword resolution.
_QUEUE_TO_CATEGORY: Final[dict[str, str]] = {
    "Billing and Payments": "Billing",
}


# `tag_1` (free-text label) gives us strong hints. Exact (lowercased) matches:
_TAG_TO_CATEGORY: Final[dict[str, str]] = {
    "billing": "Billing",
    "payment": "Billing",
    "refund": "Billing",
    "subscription": "Billing",
    "account": "Authentication",
    "login": "Authentication",
    "password": "Authentication",
    "access": "Authentication",
    "auth": "Authentication",
    "security": "Authentication",
    "bug": "Bug Report",
    "crash": "Bug Report",
    "error": "Bug Report",
    "outage": "Bug Report",
    "performance": "Bug Report",
    "feature": "Feature Request",
    "enhancement": "Feature Request",
    "integration": "Technical Setup",
    "setup": "Technical Setup",
    "installation": "Technical Setup",
    "configuration": "Technical Setup",
    "documentation": "Technical Setup",
    "onboarding": "Technical Setup",
    "api": "Technical Setup",
    "sdk": "Technical Setup",
}


def _row_text_blob(row: Mapping[str, object]) -> str:
    """Concatenate the textual fields we will scan for keyword signals."""
    parts: list[str] = []
    for col in ("subject", "body", "tag_1", "tag_2", "tag_3", "tag_4"):
        val = row.get(col)
        if isinstance(val, str) and val:
            parts.append(val.lower())
    return " \n ".join(parts)


def _keyword_category(text: str) -> str | None:
    """Return the first category whose keyword set matches the text blob, or None."""
    for kw in _BILLING_KEYWORDS:
        if kw in text:
            return "Billing"
    for kw in _AUTH_KEYWORDS:
        if kw in text:
            return "Authentication"
    for kw in _BUG_KEYWORDS:
        if kw in text:
            return "Bug Report"
    for kw in _FEATURE_KEYWORDS:
        if kw in text:
            return "Feature Request"
    for kw in _SETUP_KEYWORDS:
        if kw in text:
            return "Technical Setup"
    return None


def map_to_category(row: Mapping[str, object]) -> str | None:
    """Map a single raw row to one of the 5 target categories, or None to drop.

    Precedence:
        1. `queue` exact match in _QUEUE_TO_CATEGORY (high confidence).
        2. `tag_1` exact match in _TAG_TO_CATEGORY.
        3. Keyword scan across subject + body + tag columns.
    """
    queue = row.get("queue")
    if isinstance(queue, str) and queue in _QUEUE_TO_CATEGORY:
        return _QUEUE_TO_CATEGORY[queue]

    tag1 = row.get("tag_1")
    if isinstance(tag1, str) and tag1:
        cat = _TAG_TO_CATEGORY.get(tag1.strip().lower())
        if cat is not None:
            return cat

    text = _row_text_blob(row)
    if not text:
        return None
    return _keyword_category(text)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and assert the expected columns are present."""
    df = pd.read_csv(path)
    missing = [c for c in _RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw CSV missing required columns: {missing}")
    return df


def build_labeled_frame(
    raw_path: str | Path,
    *,
    english_only: bool = True,
) -> pd.DataFrame:
    """Load raw data, filter to English (optional), map labels, drop unmapped.

    Returns a DataFrame with columns: `text` (subject + body) and `category`.
    """
    df = load_raw(raw_path)

    if english_only and "language" in df.columns:
        df = df[df["language"] == "en"].reset_index(drop=True)

    df = df.copy()
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    df["category"] = df.apply(map_to_category, axis=1)
    df = df.dropna(subset=["category"]).reset_index(drop=True)

    df["text"] = (df["subject"].str.strip() + "\n\n" + df["body"].str.strip()).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    df["category"] = df["category"].astype("category")
    out = df[["text", "category"]].copy()
    out["category"] = out["category"].astype(str)
    return out


def stratified_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split on the `category` column.

    Test is carved out first, then val is carved out of the remaining train
    portion. Both splits are stratified by category.
    """
    if "category" not in df.columns:
        raise ValueError("DataFrame must have a 'category' column to stratify on")

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["category"],
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val["category"],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Write train/val/test CSVs under out_dir. Returns the three paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.csv"
    val_path = out / "val.csv"
    test_path = out / "test.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, val_path, test_path


def category_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Return category -> count for the labeled DataFrame."""
    return df["category"].value_counts().to_dict()