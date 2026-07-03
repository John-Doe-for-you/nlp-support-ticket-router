"""Rule-based priority scoring engine.

Public API:
    PRIORITY_LEVELS     : ordered tuple of the locked priority labels.
    CUSTOMER_PLAN_WEIGHTS : per-plan scalar weight in [0, 1].
    P1_THRESHOLD, P2_THRESHOLD : score cutoffs (locked in PROJECT_PLAN §7).
    URGENCY_HITS_CAP    : max urgency hits treated as full credit (3).
    PriorityBreakdown   : dataclass of the four weighted components.
    PriorityResult      : score + level + breakdown.
    urgency_match_score : normalize urgency-hit count into [0, 1].
    PriorityEngine      : stateless scorer; can be reused across requests.
    get_default_engine  : process-wide singleton accessor.
    score_ticket        : module-level convenience wrapper.

Design notes
------------
The locked formula from docs/PROJECT_PLAN.md §7 is:

    priority_score = (
        40 * urgency_keyword_matches
      + 30 * negative_sentiment_intensity
      + 20 * customer_plan_weight
      + 10 * category_confidence
    )

All four inputs are normalized to [0, 1] so the score lands in [0, 100]:

  * `urgency_keyword_matches`  ->  min(urgency_hits / URGENCY_HITS_CAP, 1.0)
  * `negative_sentiment_intensity` -> reuses `SentimentAnalyzer.negative_intensity`
                                      (defined on Day 9) which already blends
                                      VADER `neg` with an urgency-hits bonus.
  * `customer_plan_weight`     ->  {free: 0.0, pro: 0.5, enterprise: 1.0}
  * `category_confidence`      -> 0..1 from `CategoryClassifier.predict_with_confidence`

Mapping (locked):
    score >= 70  -> P1 (critical, immediate)
    40 <= score < 70 -> P2 (standard)
    score < 40   -> P3 (low)

The engine is intentionally stateless: callers pass a `SentimentResult` (or
its pieces) plus a category confidence and plan. This keeps the engine
trivially thread-safe and easy to unit-test without touching the trained
category model. The Day 11 inference orchestrator wires the real pieces
together; today we build and prove the formula in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ticket_router.models.sentiment import SentimentResult, SentimentScores


# Locked label set and score cutoffs (PROJECT_PLAN §7).
PRIORITY_LEVELS: tuple[str, ...] = ("P1", "P2", "P3")
P1_THRESHOLD: int = 70
P2_THRESHOLD: int = 40

# Per-plan scalar weight. Locked at 0.0/0.5/1.0 so the 20% plan component
# contributes 0, 10, or 20 to the final score.
CUSTOMER_PLAN_WEIGHTS: dict[str, float] = {
    "free": 0.0,
    "pro": 0.5,
    "enterprise": 1.0,
}

# How many urgency hits count as "full credit" for the urgency component.
# Three strong matches (e.g. "charged twice", "unacceptable", "lawsuit")
# is already a screaming ticket; anything beyond is capped to avoid a
# single ticket overwhelming the score.
URGENCY_HITS_CAP: int = 3

# Formula weights from PROJECT_PLAN §7. Public so tests can assert
# component-level behavior.
W_URGENCY: float = 40.0
W_SENTIMENT: float = 30.0
W_PLAN: float = 20.0
W_CONFIDENCE: float = 10.0


def urgency_match_score(urgency_hits: int, cap: int = URGENCY_HITS_CAP) -> float:
    """Normalize a raw urgency-hit count into the [0, 1] urgency component.

    `urgency_hits=0` -> 0.0, `>= cap` -> 1.0, linear in between. Negative
    inputs are clamped to 0.
    """
    if urgency_hits < 0:
        return 0.0
    if cap <= 0:
        return 1.0 if urgency_hits > 0 else 0.0
    return min(float(urgency_hits) / float(cap), 1.0)


def _clamp_unit(value: float) -> float:
    """Clamp `value` into [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


@dataclass(frozen=True)
class PriorityBreakdown:
    """Per-component view of the priority score, all in [0, 1].

    The sum of components, weighted by `W_*`, equals `PriorityResult.score`.
    Exposed on the result so the Day 17 latency/debug response can show
    *why* a ticket landed at P1, not just *that* it did.
    """

    urgency: float
    sentiment: float
    plan: float
    confidence: float

    def weighted_sum(self) -> float:
        return (
            W_URGENCY * self.urgency
            + W_SENTIMENT * self.sentiment
            + W_PLAN * self.plan
            + W_CONFIDENCE * self.confidence
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "urgency": float(self.urgency),
            "sentiment": float(self.sentiment),
            "plan": float(self.plan),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class PriorityResult:
    """Structured output of `PriorityEngine.score`."""

    level: str
    score: int
    breakdown: PriorityBreakdown

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "score": int(self.score),
            "breakdown": self.breakdown.to_dict(),
        }


def _level_from_score(score: float) -> str:
    """Map a continuous score in [0, 100] to one of the locked labels."""
    if score >= P1_THRESHOLD:
        return "P1"
    if score >= P2_THRESHOLD:
        return "P2"
    return "P3"


class PriorityEngine:
    """Stateless priority scorer.

    Construction is cheap (no model loading); the engine is safe to share
    across threads. For request paths, prefer `get_default_engine()`.
    """

    def __init__(
        self,
        *,
        customer_plan_weights: dict[str, float] | None = None,
        urgency_hits_cap: int = URGENCY_HITS_CAP,
    ) -> None:
        self.customer_plan_weights: dict[str, float] = (
            dict(CUSTOMER_PLAN_WEIGHTS)
            if customer_plan_weights is None
            else dict(customer_plan_weights)
        )
        self.urgency_hits_cap: int = int(urgency_hits_cap)

    def urgency_component(self, urgency_hits: int) -> float:
        return urgency_match_score(urgency_hits, cap=self.urgency_hits_cap)

    def plan_component(self, customer_plan: str) -> float:
        if customer_plan not in self.customer_plan_weights:
            raise ValueError(
                f"unknown customer_plan {customer_plan!r}; "
                f"expected one of {sorted(self.customer_plan_weights)}"
            )
        return float(self.customer_plan_weights[customer_plan])

    @staticmethod
    def _negative_intensity_from_sentiment(
        sentiment: "SentimentResult | None",
        urgency_hits: int | None,
    ) -> float:
        """Pull the [0, 1] negative-intensity scalar from a SentimentResult.

        If `urgency_hits` is given explicitly, it overrides the hit count
        on the result — handy for tests that want to drive the urgency
        component independently of the sentiment component. If neither is
        available, we fall back to 0.0 (neutral default).
        """
        if sentiment is None:
            return 0.0
        scores = sentiment.scores
        if urgency_hits is not None:
            from ticket_router.models.sentiment import SentimentScores

            scores = SentimentScores(
                neg=scores.neg,
                neu=scores.neu,
                pos=scores.pos,
                compound=scores.compound,
                urgency_hits=int(urgency_hits),
            )
        # Late import to avoid a hard module-level dep on sentiment from
        # the API/CLI paths that only need priority.
        from ticket_router.models.sentiment import get_default_analyzer

        return get_default_analyzer().negative_intensity(scores)

    def score(
        self,
        *,
        sentiment: "SentimentResult | None" = None,
        urgency_hits: int | None = None,
        category_confidence: float = 0.0,
        customer_plan: str = "free",
    ) -> PriorityResult:
        """Compute a `PriorityResult` from the four locked inputs.

        Parameters
        ----------
        sentiment
            Optional `SentimentResult` from Day 9. If provided, the
            negative-intensity component is derived from it via
            `SentimentAnalyzer.negative_intensity`. Ignored if both
            `urgency_hits` and a sentiment are given: the explicit hit
            count wins.
        urgency_hits
            Optional integer override for the urgency hit count. When
            supplied without a `sentiment`, the negative-intensity
            component falls back to 0.0.
        category_confidence
            Float in [0, 1] from the category classifier. Clamped.
        customer_plan
            One of the keys of `CUSTOMER_PLAN_WEIGHTS`.

        Raises
        ------
        ValueError
            If `customer_plan` is not a known plan.
        """
        # 1) Urgency component: hits -> [0, 1].
        hits = (
            int(urgency_hits)
            if urgency_hits is not None
            else (len(sentiment.urgency_signals) if sentiment is not None else 0)
        )
        urgency = self.urgency_component(hits)

        # 2) Sentiment component: blend VADER neg with urgency bonus.
        sentiment_value = self._negative_intensity_from_sentiment(
            sentiment,
            urgency_hits if sentiment is not None else None,
        )
        sentiment_value = _clamp_unit(sentiment_value)

        # 3) Plan component.
        plan_value = _clamp_unit(self.plan_component(customer_plan))

        # 4) Confidence component.
        confidence_value = _clamp_unit(float(category_confidence))

        breakdown = PriorityBreakdown(
            urgency=urgency,
            sentiment=sentiment_value,
            plan=plan_value,
            confidence=confidence_value,
        )
        raw_score = breakdown.weighted_sum()
        # Round to int for the API; clamp to [0, 100] defensively.
        score_int = int(round(max(0.0, min(100.0, raw_score))))
        level = _level_from_score(score_int)
        return PriorityResult(level=level, score=score_int, breakdown=breakdown)


_DEFAULT: PriorityEngine | None = None


def get_default_engine() -> PriorityEngine:
    """Return a process-wide singleton `PriorityEngine`."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PriorityEngine()
    return _DEFAULT


def score_ticket(
    *,
    sentiment: "SentimentResult | None" = None,
    urgency_hits: int | None = None,
    category_confidence: float = 0.0,
    customer_plan: str = "free",
) -> PriorityResult:
    """Convenience wrapper using the default engine."""
    return get_default_engine().score(
        sentiment=sentiment,
        urgency_hits=urgency_hits,
        category_confidence=category_confidence,
        customer_plan=customer_plan,
    )


__all__ = [
    "PRIORITY_LEVELS",
    "P1_THRESHOLD",
    "P2_THRESHOLD",
    "CUSTOMER_PLAN_WEIGHTS",
    "URGENCY_HITS_CAP",
    "W_URGENCY",
    "W_SENTIMENT",
    "W_PLAN",
    "W_CONFIDENCE",
    "PriorityBreakdown",
    "PriorityResult",
    "PriorityEngine",
    "urgency_match_score",
    "get_default_engine",
    "score_ticket",
]
