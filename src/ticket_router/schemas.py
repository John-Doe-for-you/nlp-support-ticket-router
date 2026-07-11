"""Pydantic request/response schemas for the API.

These mirror the public contract documented in ``docs/PROJECT_PLAN.md`` §3.
Keep them stable: downstream consumers (the Day 19 eval harness, the
cURL examples in ``docs/api.md``, and any future SDK) rely on the
field names and types being byte-for-byte the same.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CustomerPlan = Literal["free", "pro", "enterprise"]
Sentiment = Literal["Positive", "Neutral", "Frustrated", "Angry"]
Priority = Literal["P1", "P2", "P3"]
Category = Literal["Billing", "Authentication", "Bug Report", "Feature Request", "Technical Setup"]


class ClassifyRequest(BaseModel):
    """Single-ticket classify request.

    ``text`` is required; everything else has sensible defaults so
    curl examples in the README stay one-liners.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=8000, description="Raw support ticket text")
    customer_plan: CustomerPlan = "free"
    customer_id: str | None = None


class SentimentScores(BaseModel):
    neg: float
    neu: float
    pos: float
    compound: float


class ClassifyResponse(BaseModel):
    ticket_id: str
    category: Category
    category_confidence: float
    sentiment: Sentiment
    sentiment_scores: SentimentScores
    priority: Priority
    priority_score: int
    routed_to: str
    urgency_signals: list[str]
    latency_ms: int


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

# Hard cap on batch size. Larger requests would blow past the 100ms p99
# latency SLO (the classifier is sequential, ~15ms per ticket) and risk
# starving other clients. Anything bigger should be split client-side.
MAX_BATCH_SIZE: int = 50


class BatchClassifyRequest(BaseModel):
    """Request body for ``POST /classify/batch``.

    Accepts a list of :class:`ClassifyRequest` items. ``extra='forbid'``
    keeps typos like ``{"tickets": [...]}`` from being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[ClassifyRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class BatchClassifyResponse(BaseModel):
    """Response body for ``POST /classify/batch``.

    Mirrors :class:`ClassifyResponse` per item plus a top-level
    ``count`` and aggregate ``latency_ms`` (sum across the whole batch)
    so dashboards can plot throughput without a second round-trip.
    """

    count: int
    latency_ms: int
    results: list[ClassifyResponse]
