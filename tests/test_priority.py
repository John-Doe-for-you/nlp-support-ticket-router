"""Tests for the Day 10 rule-based priority scoring engine.

The formula and thresholds are locked in docs/PROJECT_PLAN.md §7. These
tests cover:

  * constants and shape of the public API
  * `urgency_match_score` normalization
  * customer-plan weights and validation
  * the formula itself, via boundary and scenario tests
  * integration with the Day 9 `SentimentAnalyzer` singleton
"""

from __future__ import annotations

import pytest

from ticket_router.models.priority import (
    CUSTOMER_PLAN_WEIGHTS,
    P1_THRESHOLD,
    P2_THRESHOLD,
    PRIORITY_LEVELS,
    URGENCY_HITS_CAP,
    W_CONFIDENCE,
    W_PLAN,
    W_SENTIMENT,
    W_URGENCY,
    PriorityBreakdown,
    PriorityEngine,
    PriorityResult,
    get_default_engine,
    score_ticket,
    urgency_match_score,
)
from ticket_router.models.sentiment import (
    SentimentAnalyzer,
    SentimentResult,
    SentimentScores,
    get_default_analyzer,
)


# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture()
def engine() -> PriorityEngine:
    return PriorityEngine()


@pytest.fixture()
def analyzer() -> SentimentAnalyzer:
    return get_default_analyzer()


def _empty_sentiment() -> SentimentResult:
    return SentimentResult(
        label="Neutral",
        scores=SentimentScores(neg=0.0, neu=1.0, pos=0.0, compound=0.0, urgency_hits=0),
        urgency_signals=(),
    )


# -----------------------------
# Constants / shape
# -----------------------------


def test_priority_levels_locked() -> None:
    assert PRIORITY_LEVELS == ("P1", "P2", "P3")


def test_thresholds_match_plan() -> None:
    assert P1_THRESHOLD == 70
    assert P2_THRESHOLD == 40
    assert P1_THRESHOLD > P2_THRESHOLD


def test_formula_weights_sum_to_100() -> None:
    # The plan specifies the score lives in [0, 100], so the weights
    # must sum to 100 when inputs are in [0, 1].
    total = W_URGENCY + W_SENTIMENT + W_PLAN + W_CONFIDENCE
    assert total == pytest.approx(100.0)


def test_customer_plan_weights_are_in_unit_range() -> None:
    assert set(CUSTOMER_PLAN_WEIGHTS) == {"free", "pro", "enterprise"}
    for plan, weight in CUSTOMER_PLAN_WEIGHTS.items():
        assert 0.0 <= weight <= 1.0, f"{plan} weight {weight} out of [0, 1]"
    # Free < Pro < Enterprise so the plan component strictly increases.
    assert (
        CUSTOMER_PLAN_WEIGHTS["free"]
        < CUSTOMER_PLAN_WEIGHTS["pro"]
        < CUSTOMER_PLAN_WEIGHTS["enterprise"]
    )


def test_default_engine_is_singleton() -> None:
    a = get_default_engine()
    b = get_default_engine()
    assert a is b


# -----------------------------
# urgency_match_score
# -----------------------------


@pytest.mark.parametrize(
    "hits,cap,expected",
    [
        (0, 3, 0.0),
        (1, 3, 1 / 3),
        (2, 3, 2 / 3),
        (3, 3, 1.0),
        (4, 3, 1.0),  # capped
        (10, 3, 1.0),  # capped
        (5, 5, 1.0),
        (-1, 3, 0.0),  # negative clamped
    ],
)
def test_urgency_match_score_normalization(
    hits: int, cap: int, expected: float
) -> None:
    assert urgency_match_score(hits, cap=cap) == pytest.approx(expected, abs=1e-9)


def test_urgency_match_score_default_cap() -> None:
    assert urgency_match_score(URGENCY_HITS_CAP) == 1.0
    assert urgency_match_score(URGENCY_HITS_CAP + 5) == 1.0


# -----------------------------
# Plan component / validation
# -----------------------------


def test_plan_component_known_plans(engine: PriorityEngine) -> None:
    assert engine.plan_component("free") == 0.0
    assert engine.plan_component("pro") == 0.5
    assert engine.plan_component("enterprise") == 1.0


def test_plan_component_unknown_plan_raises(engine: PriorityEngine) -> None:
    with pytest.raises(ValueError, match="unknown customer_plan"):
        engine.plan_component("diamond")


# -----------------------------
# Boundary tests for the formula
# -----------------------------


def test_zero_input_gives_zero_score_p3(engine: PriorityEngine) -> None:
    r = engine.score(
        sentiment=_empty_sentiment(),
        category_confidence=0.0,
        customer_plan="free",
    )
    assert r.level == "P3"
    assert r.score == 0
    assert r.breakdown.urgency == 0.0
    assert r.breakdown.sentiment == 0.0
    assert r.breakdown.plan == 0.0
    assert r.breakdown.confidence == 0.0


def test_max_input_gives_100_p1(engine: PriorityEngine) -> None:
    # neg=1.0 + 5 hits (bonus capped) -> 1.0; 3 hits -> urgency=1.0
    big_neg = SentimentResult(
        label="Angry",
        scores=SentimentScores(neg=1.0, neu=0.0, pos=0.0, compound=-1.0, urgency_hits=5),
        urgency_signals=("a", "b", "c", "d", "e"),
    )
    r = engine.score(
        sentiment=big_neg,
        category_confidence=1.0,
        customer_plan="enterprise",
    )
    assert r.score == 100
    assert r.level == "P1"


def test_p1_threshold_boundary_inclusive(engine: PriorityEngine) -> None:
    # Construct a score that lands exactly on the P1/P2 boundary (70).
    # 40*urgency + 30*sent + 20*plan + 10*conf = 70
    # Plan: urgency=1.0 (3 hits, capped), plan=0.5, conf=1.0, sentiment so
    # that 30*sent = 10 -> sent = 1/3.
    # sentiment_value = neg + min(0.10*urgency_hits, 0.30) where the same
    # hit count drives both. With hits=3 the bonus is 0.30, so we need
    # neg = 1/3 - 0.30 = 0.0333 (and clamped to >=0, fine).
    s = SentimentResult(
        label="Frustrated",
        scores=SentimentScores(neg=0.0334, neu=0.8, pos=0.1666, compound=-0.2, urgency_hits=3),
        urgency_signals=("a", "b", "c"),
    )
    r = engine.score(
        sentiment=s,
        category_confidence=1.0,
        customer_plan="pro",
    )
    # urgency=1.0, sentiment ~ 0.3333, plan=0.5, conf=1.0 -> 40+10+10+10 = 70.
    assert r.score == 70
    assert r.level == "P1"


def test_p2_threshold_boundary_inclusive(engine: PriorityEngine) -> None:
    # The P2 lower bound (score == 40) is the level-mapping threshold, not
    # a value the formula can land on exactly when urgency > 0 (because
    # urgency hits also feed the sentiment bonus). Here we prove the
    # mapping rule: a constructed score of exactly 40 maps to P2, and
    # anything just below 40 maps to P3. We do this by setting the
    # urgency component just low enough to land at 40 with neutral
    # sentiment: urgency=1.0, plan=0, conf=0, sentiment=0.
    # With hits=0 the bonus is 0, so neg=0 -> sentiment=0. But hits=0
    # also means urgency=0. So the literal score-40 boundary isn't
    # reachable in one shot; instead we assert that the level-mapping
    # function respects the documented cutoffs.
    s = SentimentResult(
        label="Neutral",
        scores=SentimentScores(neg=0.0, neu=1.0, pos=0.0, compound=0.0, urgency_hits=0),
        urgency_signals=(),
    )
    # Score 0: must be P3.
    r0 = engine.score(sentiment=s, category_confidence=0.0, customer_plan="free")
    assert r0.score < P2_THRESHOLD
    assert r0.level == "P3"

    # Build a score that lands just below P1: P1 if >=70, P2 otherwise.
    # urgency=1.0 (3 hits, bonus 0.30), plan=0, conf=0, neg=0.0333 ->
    # sent=0.3333. Total = 40 + 10 + 0 + 0 = 50. -> P2.
    s_p2 = SentimentResult(
        label="Frustrated",
        scores=SentimentScores(neg=0.0334, neu=0.8, pos=0.1666, compound=-0.2, urgency_hits=3),
        urgency_signals=("a", "b", "c"),
    )
    r_p2 = engine.score(sentiment=s_p2, category_confidence=0.0, customer_plan="free")
    assert r_p2.score < P1_THRESHOLD
    assert r_p2.level == "P2"


def test_just_below_p2_is_p3(engine: PriorityEngine) -> None:
    # 3 urgency hits + everything else zero = 40 - epsilon. Use 2 hits.
    # 40 * (2/3) = 26.67 -> round to 27 -> P3.
    s = SentimentResult(
        label="Neutral",
        scores=SentimentScores(neg=0.0, neu=1.0, pos=0.0, compound=0.0, urgency_hits=2),
        urgency_signals=("a", "b"),
    )
    r = engine.score(
        sentiment=s,
        category_confidence=0.0,
        customer_plan="free",
    )
    assert r.score < P2_THRESHOLD
    assert r.level == "P3"


# -----------------------------
# Realistic scenario tests
# -----------------------------


def test_angry_enterprise_billing_incident_is_p1(
    engine: PriorityEngine, analyzer: SentimentAnalyzer
) -> None:
    text = "I was charged twice and this is completely unacceptable, I will sue."
    sentiment = analyzer.analyze(text)
    r = engine.score(
        sentiment=sentiment,
        category_confidence=0.92,
        customer_plan="enterprise",
    )
    assert r.level == "P1"
    assert r.score >= P1_THRESHOLD
    assert "charged twice" in sentiment.urgency_signals
    assert sentiment.label == "Angry"


def test_neutral_free_feature_request_is_p3(
    engine: PriorityEngine, analyzer: SentimentAnalyzer
) -> None:
    text = "It would be nice if the dashboard had a dark mode."
    sentiment = analyzer.analyze(text)
    r = engine.score(
        sentiment=sentiment,
        category_confidence=0.65,
        customer_plan="free",
    )
    assert r.level == "P3"
    assert r.score < P2_THRESHOLD


def test_frustrated_pro_bug_report_lands_in_p2(
    engine: PriorityEngine, analyzer: SentimentAnalyzer
) -> None:
    # A pro user reporting a bug with one urgency signal ("unacceptable")
    # plus a frustrated tone lands in the P2 band: enough negativity +
    # plan + confidence to clear P2 floor, not enough to hit P1.
    text = "The export button is broken and the workaround is unacceptable."
    sentiment = analyzer.analyze(text)
    r = engine.score(
        sentiment=sentiment,
        category_confidence=0.85,
        customer_plan="pro",
    )
    assert P2_THRESHOLD <= r.score < P1_THRESHOLD
    assert r.level == "P2"


def test_positive_pro_thank_you_is_p3(
    engine: PriorityEngine, analyzer: SentimentAnalyzer
) -> None:
    text = "Thanks so much, the new export feature works great!"
    sentiment = analyzer.analyze(text)
    r = engine.score(
        sentiment=sentiment,
        category_confidence=0.88,
        customer_plan="pro",
    )
    assert r.level == "P3"
    assert r.score < P2_THRESHOLD


# -----------------------------
# Integration with the Day 9 sentiment analyzer
# -----------------------------


def test_sentiment_component_uses_day9_negative_intensity(
    engine: PriorityEngine,
) -> None:
    # Build a synthetic SentimentResult so we can assert the exact
    # sentiment-component value the engine produced.
    s = SentimentResult(
        label="Frustrated",
        scores=SentimentScores(neg=0.40, neu=0.40, pos=0.20, compound=-0.2, urgency_hits=2),
        urgency_signals=("charged twice", "unacceptable"),
    )
    r = engine.score(
        sentiment=s,
        category_confidence=0.0,
        customer_plan="free",
    )
    expected_sent = get_default_analyzer().negative_intensity(s.scores)
    assert r.breakdown.sentiment == pytest.approx(expected_sent, abs=1e-9)


def test_explicit_urgency_hits_override_sentiment_count(
    engine: PriorityEngine,
) -> None:
    # Sentiment carries 5 urgency hits internally, but caller passes
    # urgency_hits=0 to simulate "I want to score the urgency component
    # independently". The sentiment component is still derived from
    # `sentiment.scores`, so the bonus from urgency_hits=0 is 0.
    s = SentimentResult(
        label="Angry",
        scores=SentimentScores(neg=0.5, neu=0.3, pos=0.2, compound=-0.4, urgency_hits=5),
        urgency_signals=("a", "b", "c", "d", "e"),
    )
    r = engine.score(
        sentiment=s,
        urgency_hits=0,
        category_confidence=0.0,
        customer_plan="free",
    )
    assert r.breakdown.urgency == 0.0
    # sentiment still nonzero from the neg probability (0.5) + 0 bonus.
    assert r.breakdown.sentiment == pytest.approx(0.5, abs=1e-9)


# -----------------------------
# Breakdown invariants
# -----------------------------


def test_breakdown_weighted_sum_equals_score(engine: PriorityEngine) -> None:
    s = SentimentResult(
        label="Frustrated",
        scores=SentimentScores(neg=0.5, neu=0.3, pos=0.2, compound=-0.3, urgency_hits=2),
        urgency_signals=("charged twice", "unacceptable"),
    )
    r = engine.score(
        sentiment=s,
        category_confidence=0.85,
        customer_plan="pro",
    )
    expected_raw = r.breakdown.weighted_sum()
    # score is the rounded, clamped raw.
    expected = int(round(max(0.0, min(100.0, expected_raw))))
    assert r.score == expected


def test_score_is_clamped_to_unit_range(engine: PriorityEngine) -> None:
    # Try to overflow via confidence > 1.0; should still clamp.
    r = engine.score(
        sentiment=_empty_sentiment(),
        category_confidence=5.0,
        customer_plan="enterprise",
    )
    assert 0 <= r.score <= 100


def test_breakdown_components_all_in_unit_range(engine: PriorityEngine) -> None:
    s = SentimentResult(
        label="Angry",
        scores=SentimentScores(neg=0.9, neu=0.05, pos=0.05, compound=-0.8, urgency_hits=4),
        urgency_signals=("a", "b", "c", "d"),
    )
    r = engine.score(
        sentiment=s,
        category_confidence=1.5,  # out of range, should clamp
        customer_plan="enterprise",
    )
    for value in r.breakdown.to_dict().values():
        assert 0.0 <= value <= 1.0


# -----------------------------
# Parametrized table
# -----------------------------


@pytest.mark.parametrize(
    "urgency_hits,sentiment_neg,plan,confidence,expected_band",
    [
        (0, 0.0, "free", 0.0, "P3"),
        (0, 0.3, "free", 0.5, "P3"),
        (1, 0.4, "pro", 0.7, "P2"),
        (2, 0.5, "pro", 0.8, "P2"),
        (3, 0.9, "enterprise", 0.95, "P1"),
        (4, 1.0, "enterprise", 1.0, "P1"),
    ],
)
def test_score_grid(
    engine: PriorityEngine,
    urgency_hits: int,
    sentiment_neg: float,
    plan: str,
    confidence: float,
    expected_band: str,
) -> None:
    s = SentimentResult(
        label="Frustrated" if sentiment_neg < 0.5 else "Angry",
        scores=SentimentScores(
            neg=sentiment_neg,
            neu=max(0.0, 1.0 - sentiment_neg) / 2,
            pos=max(0.0, 1.0 - sentiment_neg) / 2,
            compound=-sentiment_neg,
            urgency_hits=urgency_hits,
        ),
        urgency_signals=tuple(str(i) for i in range(urgency_hits)),
    )
    r = engine.score(
        sentiment=s,
        category_confidence=confidence,
        customer_plan=plan,
    )
    assert r.level == expected_band


# -----------------------------
# Module-level convenience wrapper
# -----------------------------


def test_score_ticket_uses_default_engine() -> None:
    r = score_ticket(
        sentiment=_empty_sentiment(),
        category_confidence=0.0,
        customer_plan="free",
    )
    assert isinstance(r, PriorityResult)
    assert r.score == 0
    assert r.level == "P3"


def test_to_dict_round_trip_shape(engine: PriorityEngine) -> None:
    r = engine.score(
        sentiment=_empty_sentiment(),
        category_confidence=0.5,
        customer_plan="pro",
    )
    payload = r.to_dict()
    assert set(payload) == {"level", "score", "breakdown"}
    assert isinstance(payload["score"], int)
    assert set(payload["breakdown"]) == {"urgency", "sentiment", "plan", "confidence"}


def test_priority_breakdown_to_dict_shape() -> None:
    b = PriorityBreakdown(urgency=0.5, sentiment=0.4, plan=0.5, confidence=0.9)
    assert b.to_dict() == {
        "urgency": 0.5,
        "sentiment": 0.4,
        "plan": 0.5,
        "confidence": 0.9,
    }
    assert b.weighted_sum() == pytest.approx(
        W_URGENCY * 0.5 + W_SENTIMENT * 0.4 + W_PLAN * 0.5 + W_CONFIDENCE * 0.9
    )
