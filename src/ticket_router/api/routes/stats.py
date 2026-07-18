"""``GET /stats`` and ``GET /stats/{facet}`` aggregation routes (Day 16).

Contracts
---------

::

    GET /stats
    ->
    {
      "total_predictions": <int>,
      "total_tickets":     <int>,
      "by_category":   { "total": <int>, "items": [ {"Billing": 12}, ... ] },
      "by_sentiment":  { "total": <int>, "items": [ {"Angry":   3}, ... ] },
      "by_priority":   { "total": <int>, "items": [ {"P1":      5}, ... ] },
      "by_team":       { "total": <int>, "items": [ {"billing-team": 7}, ... ] },
    }

::

    GET /stats/{facet}        # facet in {category, sentiment, priority, team}
    ->
    { "facet": "...", "total": <int>, "items": [ {...}, ... ] }

Design notes
------------

* Every facet is a single ``GROUP BY`` over the ``predictions`` table
  using the existing Day 13 ``count_by_column`` helper, so we get the
  benefits of the indexes added on Day 13 (``category``,
  ``priority``, ``routed_to``) for free. Sentiment and ``routed_to``
  (team) reuse the same code path - we just point ``count_by_column``
  at a different column attribute.
* Each ``items`` entry is a single-key dict so the wire format
  mirrors ``{value: count}`` and clients can iterate without knowing
  which column produced the breakdown.
* ``total_predictions`` and ``total_tickets`` are independent so a
  dashboard can show "120 tickets, 134 predictions" when re-runs
  have happened.
* The endpoint is read-only and never returns 503 - empty data is
  reported as ``"items": []``, not an error.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ticket_router.db import (
    count_by_column,
    count_predictions,
    count_tickets,
)
from ticket_router.db.database import get_db
from ticket_router.db.models import Prediction
from ticket_router.schemas import StatsBreakdown, StatsFacetResponse, StatsResponse

logger = logging.getLogger("ticket_router.api.stats")

router = APIRouter(prefix="/stats", tags=["stats"])


# ---------------------------------------------------------------------------
# Facet -> ORM column mapping
# ---------------------------------------------------------------------------


# Locked set of valid facet names. We accept ``str`` (not ``Literal``)
# at the route layer because FastAPI would translate an unknown value
# into a 422 (request-validation) error. The Day 16 contract requires
# 400 because the URL is well-formed - the *value* is the problem -
# so we validate manually and raise :class:`HTTPException` ourselves.
_FACET_TO_COLUMN: dict[str, Any] = {
    "category": Prediction.category,
    "sentiment": Prediction.sentiment,
    "priority": Prediction.priority,
    "team": Prediction.routed_to,
}


def _breakdown_from_counts(counts: dict[str, int]) -> StatsBreakdown:
    """Wrap a ``{value: count}`` dict in the public ``StatsBreakdown`` shape.

    The repository already sorts by descending count, so we keep the
    order. We also compute ``total`` as the sum (so a partial-DB
    scenario where a facet was added after the count can still show
    a sensible total).
    """
    items = [{k: v} for k, v in counts.items()]
    total = sum(counts.values())
    return StatsBreakdown(total=total, items=items)


def _facet_response(facet: str, counts: dict[str, int]) -> StatsFacetResponse:
    return StatsFacetResponse(
        facet=facet,
        total=sum(counts.values()),
        items=[{k: v} for k, v in counts.items()],
    )


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Aggregated counts for category, sentiment, priority, and team",
)
def get_stats(session: Session = Depends(get_db)) -> StatsResponse:
    """Return the four core facet breakdowns plus table-wide totals.

    Designed to be the single round-trip a dashboard needs: four
    GROUP BY queries (the ORM's identity-map cache means the
    table-wide ``count(*)` calls reuse the same connection). For a
    real Postgres deployment we'd batch them into one CTE, but SQLite
    + the in-memory test path don't benefit, and the four-queries
    version is much easier to reason about.
    """
    by_category = _breakdown_from_counts(count_by_column(session, Prediction.category))
    by_sentiment = _breakdown_from_counts(count_by_column(session, Prediction.sentiment))
    by_priority = _breakdown_from_counts(count_by_column(session, Prediction.priority))
    by_team = _breakdown_from_counts(count_by_column(session, Prediction.routed_to))

    return StatsResponse(
        total_predictions=count_predictions(session),
        total_tickets=count_tickets(session),
        by_category=by_category,
        by_sentiment=by_sentiment,
        by_priority=by_priority,
        by_team=by_team,
    )


# ---------------------------------------------------------------------------
# GET /stats/{facet}
# ---------------------------------------------------------------------------


@router.get(
    "/{facet}",
    response_model=StatsFacetResponse,
    status_code=status.HTTP_200_OK,
    summary="Count breakdown for a single facet",
    responses={status.HTTP_400_BAD_REQUEST: {"description": "Unknown facet"}},
)
def get_stats_facet(
    facet: str,
    session: Session = Depends(get_db),
) -> StatsFacetResponse:
    """Return one facet's breakdown. ``facet`` is one of:
    ``category``, ``sentiment``, ``priority``, ``team``.

    Unknown facets return 400 (not 404) because the URL is well-formed;
    the *value* is the problem, which is a client error. We accept
    ``str`` (not ``Literal``) so we can produce a clean 400 with our
    own error message instead of FastAPI's default 422 validation
    error.
    """
    column = _FACET_TO_COLUMN.get(facet)
    if column is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown facet {facet!r}. "
                f"Expected one of: {sorted(_FACET_TO_COLUMN)}"
            ),
        )
    counts = count_by_column(session, column)
    return _facet_response(facet, counts)


__all__ = ["router"]
