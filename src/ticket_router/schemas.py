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


# ---------------------------------------------------------------------------
# Ticket history (Day 16)
# ---------------------------------------------------------------------------


# Hard cap on the ``limit`` query param for ``GET /tickets`` and friends.
# Higher values risk starving the SQLite connection under load, so we
# clamp the param and let the caller paginate.
MAX_TICKETS_LIMIT: int = 200


class TicketPrediction(BaseModel):
    """One :class:`Prediction` row associated with a ticket.

    Includes the full public ``ClassifyResponse``-shaped fields plus the
    raw ``sentiment_scores`` broken out into their component numbers
    (neg / neu / pos / compound) for dashboards.
    """

    id: int
    category: Category
    category_confidence: float
    sentiment: Sentiment
    sentiment_scores: SentimentScores
    priority: Priority
    priority_score: int
    routed_to: str
    urgency_signals: list[str]
    latency_ms: int
    created_at: str


class TicketHistoryItem(BaseModel):
    """One ticket in the history list, with its latest prediction embedded.

    The history endpoint always returns the *latest* prediction per
    ticket (re-runs create additional rows; we don't surface those in
    the list to keep the payload small). The full audit trail is
    available via :class:`TicketDetail`.
    """

    ticket_id: str
    customer_id: str | None
    customer_plan: str
    text_preview: str
    text_length: int
    created_at: str
    latest_prediction: TicketPrediction | None


class TicketHistoryResponse(BaseModel):
    """Response body for ``GET /tickets``.

    ``total`` is the total row count in the ``tickets`` table (not the
    page size) so the caller can compute whether more pages remain.
    """

    count: int
    total: int
    limit: int
    offset: int
    items: list[TicketHistoryItem]


class TicketDetail(BaseModel):
    """Full ticket detail: input + every prediction in audit-trail order.

    ``predictions`` is ordered newest-first to match the rest of the
    API. ``text`` is the original raw input; ``text_preview`` is the
    same thing truncated for list rendering.
    """

    ticket_id: str
    customer_id: str | None
    customer_plan: str
    text: str
    created_at: str
    predictions: list[TicketPrediction]


# ---------------------------------------------------------------------------
# Stats (Day 16)
# ---------------------------------------------------------------------------


class StatsBreakdown(BaseModel):
    """A single ``{label: count}`` breakdown.

    Returned by ``GET /stats`` for each facet (category, sentiment,
    priority, team). Items are sorted by descending count to match the
    repository's ``count_by_column`` ordering, so dashboards can render
    the most common values first.
    """

    total: int
    items: list[dict[str, int]]


class StatsResponse(BaseModel):
    """Response body for ``GET /stats``.

    Four facets in one round-trip so a dashboard can render four
    bar-charts without N+1 calls. The shape is intentionally flat: a
    ``/stats/{facet}`` endpoint is also exposed for callers that only
    need one dimension.
    """

    total_predictions: int
    total_tickets: int
    by_category: StatsBreakdown
    by_sentiment: StatsBreakdown
    by_priority: StatsBreakdown
    by_team: StatsBreakdown


class StatsFacetResponse(BaseModel):
    """Response body for ``GET /stats/{facet}``.

    Same shape as the ``StatsBreakdown`` entries from
    :class:`StatsResponse`, lifted to a top-level model so it can be
    documented independently in OpenAPI.
    """

    facet: str
    total: int
    items: list[dict[str, int]]
