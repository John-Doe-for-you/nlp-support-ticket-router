"""Text cleaning utilities. Implemented on Day 4."""

from __future__ import annotations

import re

_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/emails, collapse whitespace.

    Implementation is finalized on Day 4. Placeholder keeps the import working
    so other modules can be wired up incrementally.
    """
    text = text.lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text
