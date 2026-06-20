"""Text cleaning utilities for support ticket NLP.

Public API:
    clean_text(text) -> str
        Lowercase, strip HTML / URLs / emails, collapse whitespace.
        Raises TypeError if input is not a str.
"""

from __future__ import annotations

import re

_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
_WS_RE = re.compile(r"\s+")


def remove_html(text: str) -> str:
    return _HTML_RE.sub(" ", text)


def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def remove_emails(text: str) -> str:
    return _EMAIL_RE.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"clean_text expects str, got {type(text).__name__}")
    text = text.lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = normalize_whitespace(text)
    return text