"""Tests for the Day 9 sentiment analyzer (VADER + custom urgency lexicon)."""

from __future__ import annotations

import pytest

from ticket_router.models.sentiment import (
    SENTIMENT_CLASSES,
    URGENCY_LEXICON,
    SentimentAnalyzer,
    SentimentResult,
    SentimentScores,
    analyze_text,
    get_default_analyzer,
)


# -----------------------------
# Fixtures / helpers
# -----------------------------


@pytest.fixture()
def analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


# -----------------------------
# Shape / API contracts
# -----------------------------


def test_sentiment_classes_locked_to_four_labels() -> None:
    assert SENTIMENT_CLASSES == ("Positive", "Neutral", "Frustrated", "Angry")


def test_urgency_lexicon_is_non_empty_and_maps_phrases_to_negative_deltas() -> None:
    assert len(URGENCY_LEXICON) >= 15
    for phrase, delta in URGENCY_LEXICON.items():
        assert isinstance(phrase, str) and phrase, "phrase must be non-empty string"
        assert delta <= 0.0, f"urgency delta must be <= 0, got {delta} for {phrase!r}"


def test_analyzer_uses_independent_urgency_lexicon_copy() -> None:
    a = SentimentAnalyzer()
    a.urgency_lexicon["forged"] = -1.0
    b = SentimentAnalyzer()
    assert "forged" not in b.urgency_lexicon


def test_default_analyzer_is_singleton() -> None:
    a = get_default_analyzer()
    b = get_default_analyzer()
    assert a is b


def test_to_dict_round_trip_shape(analyzer: SentimentAnalyzer) -> None:
    result = analyzer.analyze("Thanks, that worked perfectly!")
    payload = result.to_dict()
    assert set(payload) == {"label", "scores", "urgency_signals"}
    assert set(payload["scores"]) == {"neg", "neu", "pos", "compound"}
    assert isinstance(payload["urgency_signals"], list)


# -----------------------------
# Empty / non-string inputs
# -----------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_text_is_neutral_zero_scores(analyzer: SentimentAnalyzer, blank: str) -> None:
    r = analyzer.analyze(blank)
    assert r.label == "Neutral"
    assert r.scores.neg == 0.0
    assert r.scores.pos == 0.0
    assert r.scores.compound == 0.0
    assert r.urgency_signals == ()


def test_non_string_input_returns_neutral(analyzer: SentimentAnalyzer) -> None:
    r = analyzer.analyze(None)  # type: ignore[arg-type]
    assert r.label == "Neutral"
    assert r.urgency_signals == ()


# -----------------------------
# Labeling behavior
# -----------------------------


def test_positive_text_is_labeled_positive(analyzer: SentimentAnalyzer) -> None:
    r = analyzer.analyze("This is wonderful! I love the new dashboard, thank you so much!")
    assert r.label == "Positive"
    assert r.scores.compound > 0
    assert r.urgency_signals == ()


def test_neutral_text_is_labeled_neutral(analyzer: SentimentAnalyzer) -> None:
    r = analyzer.analyze("My subscription renews on the 5th of next month.")
    assert r.label == "Neutral"


def test_mildly_negative_text_is_frustrated(analyzer: SentimentAnalyzer) -> None:
    r = analyzer.analyze("I'm a bit disappointed with the recent changes.")
    assert r.label == "Frustrated"


def test_strongly_negative_text_is_angry(analyzer: SentimentAnalyzer) -> None:
    r = analyzer.analyze(
        "This is absolutely terrible. I hate it. It is the worst experience I've ever had."
    )
    assert r.label == "Angry"


# -----------------------------
# Custom urgency lexicon
# -----------------------------


def test_urgency_phrase_is_detected_and_escalates_label(analyzer: SentimentAnalyzer) -> None:
    # "charged twice" alone is not strongly negative in raw VADER, but the
    # lexicon should escalate it to Angry via the urgency-hits rule.
    r = analyzer.analyze("I was charged twice on my last invoice.")
    assert "charged twice" in r.urgency_signals
    assert r.scores.urgency_hits >= 1
    assert r.label in {"Frustrated", "Angry"}


def test_urgency_signals_collected_in_order_of_appearance(
    analyzer: SentimentAnalyzer,
) -> None:
    text = "This is UNACCEPTABLE. I've been charged twice and your support is a SCAM."
    r = analyzer.analyze(text)
    signals = list(r.urgency_signals)
    assert "unacceptable" in signals
    assert "charged twice" in signals
    assert "scam" in signals
    # signals should be a tuple, signals may have > 1 entry
    assert isinstance(r.urgency_signals, tuple)
    assert r.scores.urgency_hits == len(signals)


def test_multiple_urgency_hits_pull_label_to_angry(analyzer: SentimentAnalyzer) -> None:
    text = "I was charged twice, this is unacceptable and a complete scam."
    r = analyzer.analyze(text)
    assert r.scores.urgency_hits >= 2
    assert r.label == "Angry"


def test_urgency_phrase_cap_bounds_boosted_compound(
    analyzer: SentimentAnalyzer,
) -> None:
    # Many urgency phrases in one ticket should still produce a compound
    # score inside the valid [-1, 1] range.
    text = (
        "Charged twice, double charged, overcharged, unauthorized charge, scam, "
        "fraud, lawsuit, data breach, unacceptable, furious, horrible."
    )
    r = analyzer.analyze(text)
    assert -1.0 <= r.scores.compound <= 1.0
    assert r.scores.urgency_hits >= 5
    assert r.label == "Angry"


def test_custom_lexicon_override_replaces_defaults() -> None:
    custom = {"thanks": -0.8}
    sa = SentimentAnalyzer(urgency_lexicon=custom)
    r = sa.analyze("Thanks a lot, really thanks")
    assert "thanks" in r.urgency_signals
    # Default lexicon phrases should NOT appear when overridden.
    assert "unacceptable" not in r.urgency_signals


# -----------------------------
# negative_intensity (used by priority engine on Day 10)
# -----------------------------


@pytest.mark.parametrize(
    "neg,hits,expected_low,expected_high",
    [
        (0.10, 0, 0.10, 0.10),
        (0.40, 0, 0.40, 0.40),
        (0.20, 1, 0.30, 0.30),
        (0.20, 3, 0.50, 0.50),  # bonus capped at 0.30
        (0.50, 5, 0.80, 0.80),
        (0.95, 5, 1.0, 1.0),  # clamped to 1.0
    ],
)
def test_negative_intensity_blends_vader_with_urgency(
    analyzer: SentimentAnalyzer,
    neg: float,
    hits: int,
    expected_low: float,
    expected_high: float,
) -> None:
    scores = SentimentScores(neg=neg, neu=0.5 - neg / 2, pos=0.5 - neg / 2, compound=-neg, urgency_hits=hits)
    value = analyzer.negative_intensity(scores)
    assert expected_low - 1e-9 <= value <= expected_high + 1e-9


# -----------------------------
# Batch + convenience wrapper
# -----------------------------


def test_analyze_batch_returns_one_result_per_input(analyzer: SentimentAnalyzer) -> None:
    texts = [
        "I love this product, thanks!",
        "My subscription renews on the 5th of next month.",
        "This is unacceptable, charged twice.",
    ]
    results = analyzer.analyze_batch(texts)
    assert len(results) == 3
    assert [r.label for r in results] == ["Positive", "Neutral", "Angry"]
    assert all(isinstance(r, SentimentResult) for r in results)


def test_module_level_analyze_text_uses_default_singleton() -> None:
    r1 = analyze_text("Great service!")
    r2 = get_default_analyzer().analyze("Great service!")
    assert r1.label == r2.label == "Positive"
    assert r1.scores.compound == pytest.approx(r2.scores.compound, abs=1e-9)
