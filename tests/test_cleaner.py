"""Tests for preprocessing.cleaner."""

from __future__ import annotations

import pytest

from ticket_router.preprocessing.cleaner import (
    clean_text,
    normalize_whitespace,
    remove_emails,
    remove_html,
    remove_urls,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Hello World", "hello world"),
        ("  EXTRA   spaces  ", "extra spaces"),
        ("<b>bold</b> text", "bold text"),
        ("see https://x.com/y here", "see here"),
        ("www.example.com link", "link"),
        ("mail me at foo@bar.com pls", "mail me at pls"),
        ("UPPER lower MiXeD", "upper lower mixed"),
        ("", ""),
        ("   \n\t  ", ""),
        ("line1\nline2\tline3", "line1 line2 line3"),
        ("<a href='x'>click</a>", "click"),
        ("Email: A.B-c@sub.example.co.uk!", "email: !"),
        ("multiple    spaces   between", "multiple spaces between"),
        ("HTTP://EXAMPLE.com/PATH", ""),
        ("<script>alert(1)</script>safe", "alert(1) safe"),
        ("Émoji 😊 and unicode", "émoji 😊 and unicode"),
        ("  Hello   World  ", "hello world"),
        ("Visit http://a.io and www.b.io now", "visit and now"),
        ("plain text only", "plain text only"),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_idempotent():
    s = "Hello   <b>World</b>  Visit https://x.com  and  foo@bar.com!"
    once = clean_text(s)
    twice = clean_text(once)
    assert once == twice


def test_clean_text_none_raises_typeerror():
    with pytest.raises(TypeError):
        clean_text(None)


def test_clean_text_int_raises_typeerror():
    with pytest.raises(TypeError):
        clean_text(123)


def test_remove_html_only():
    assert remove_html("<div>x</div>") == " x "


def test_remove_urls_only():
    assert remove_urls("go to https://a.io now") == "go to   now"


def test_remove_emails_only():
    assert remove_emails("hi a@b.co there") == "hi   there"


def test_normalize_whitespace_only():
    assert normalize_whitespace("  a\n\nb\tc  ") == "a b c"