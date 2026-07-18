"""``GET /tickets`` and ``GET /tickets/{ticket_id}`` history routes (Day 16).

Contracts
---------

::

    GET /tickets?limit=50&offset=0
    ->
    {
      "count": <int>,        # items in this page
      "total": <int>,        # total tickets in the table
      "limit": <int>,
      "offset": <int>,
      "items": [ TicketHistoryItem, ... ]   # newest first
    }

::

    GET /tickets/{ticket_id}
    ->
    TicketDetail { ticket_id, customer_id, customer_plan, text, created_at,
                   predictions: [ TicketPrediction, ... ] }   # newest first

Both endpoints are read-only and require only a working DB (no
category model). They will return an empty page / 404 if no data is
present, never 503, so a freshly-deployed API is still observable.

Design notes
------------

* Pagination uses ``limit`` + ``offset`` query params. The repo clamps
  ``limit`` to ``MAX_TICKETS_LIMIT`` (200) so a buggy caller cannot
  scan the whole table in one shot.
* The list view embeds *only the latest* prediction per ticket to
  keep payloads small. Re-runs (the same ``ticket_id`` getting a new
  prediction row) are preserved in the full detail endpoint.
* ``text_preview`` is the first 160 characters of the raw text, with
  a trailing ellipsis marker only when truncated. This keeps the
  typical history page under a few KB even for long pastes.
* Timestamps are serialized as ISO 8601 strings (UTC) to match the
  Pydantic-friendly contract; the underlying ``DateTime`` is naive
  UTC per the Day 13 schema decision.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ticket_router.db import (
    Prediction,
    Ticket,
    count_tickets,
    get_ticket,
    list_tickets,
)
from ticket_router.db.database import get_db
from ticket_router.schemas import (
    MAX_TICKETS_LIMIT,
    TicketDetail,
    TicketHistoryItem,
    TicketHistoryResponse,
    TicketPrediction,
)

logger = logging.getLogger("ticket_router.api.tickets")

router = APIRouter(prefix="/tickets", tags=["tickets"])


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

# Cap for the ``text_preview`` field in the list view. Picked so that
# ~2x the longest expected single-line complaint fits comfortably; a
# 160-char preview keeps each row under ~300 bytes JSON-encoded.
_TEXT_PREVIEW_LIMIT: int = 160


def _prediction_to_response(pred: Prediction) -> TicketPrediction:
    """Convert a :class:`Prediction` row to a :class:`TicketPrediction`."""
    return TicketPrediction(
        id=pred.id,
        category=pred.category,  # type: ignore[arg-type]
        category_confidence=float(pred.category_confidence),
        sentiment=pred.sentiment,  # type: ignore[arg-type]
        sentiment_scores={
            "neg": float(pred.sentiment_neg),
            "neu": float(pred.sentiment_neu),
            "pos": float(pred.sentiment_pos),
            "compound": float(pred.sentiment_compound),
        },
        priority=pred.priority,  # type: ignore[arg-type]
        priority_score=int(pred.priority_score),
        routed_to=pred.routed_to,
        urgency_signals=[s for s in (pred.urgency_signals or "").split("\n") if s],
        latency_ms=int(pred.latency_ms),
        created_at=pred.created_at.isoformat() if pred.created_at else "",
    )


def _ticket_history_item(ticket: Ticket) -> TicketHistoryItem:
    """Build a :class:`TicketHistoryItem` from a :class:`Ticket`.

    Embeds the *latest* prediction (the ``predictions`` relationship
    is already sorted newest-first by the Day 13 ORM mapping).
    """
    latest = ticket.predictions[0] if ticket.predictions else None
    text = ticket.text or ""
    if len(text) > _TEXT_PREVIEW_LIMIT:
        preview = text[: _TEXT_PREVIEW_LIMIT - 1] + "\u2026"
    else:
        preview = text
    return TicketHistoryItem(
        ticket_id=ticket.ticket_id,
        customer_id=ticket.customer_id,
        customer_plan=ticket.customer_plan,
        text_preview=preview,
        text_length=len(text),
        created_at=ticket.created_at.isoformat() if ticket.created_at else "",
        latest_prediction=_prediction_to_response(latest) if latest is not None else None,
    )


# ---------------------------------------------------------------------------
# GET /tickets
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=TicketHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="List recent tickets, newest first (paginated)",
)
def list_tickets_history(
    limit: int = Query(50, ge=1, le=MAX_TICKETS_LIMIT, description="Max items per page (1-200)"),
    offset: int = Query(0, ge=0, description="Items to skip from the start"),
    session: Session = Depends(get_db),
) -> TicketHistoryResponse:
    """Return a page of ticket history, newest-first.

    ``total`` is the table-wide count (not the page size) so the
    caller can compute ``has_more = offset + count < total``.
    """
    clamped_limit = min(int(limit), MAX_TICKETS_LIMIT)
    clamped_offset = max(int(offset), 0)
    tickets = list_tickets(session, limit=clamped_limit, offset=clamped_offset)
    total = count_tickets(session)
    return TicketHistoryResponse(
        count=len(tickets),
        total=total,
        limit=clamped_limit,
        offset=clamped_offset,
        items=[_ticket_history_item(t) for t in tickets],
    )


# ---------------------------------------------------------------------------
# GET /tickets/{ticket_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{ticket_id}",
    response_model=TicketDetail,
    status_code=status.HTTP_200_OK,
    summary="Get a single ticket with all of its predictions",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Ticket not found"}},
)
def get_ticket_detail(
    ticket_id: str,
    session: Session = Depends(get_db),
) -> TicketDetail:
    """Return one ticket and its full prediction history, newest first.

    Returns 404 when the id is unknown. We deliberately do **not** do
    a "did you mean" search: the caller knows the id format and a
    fuzzy match would make the endpoint harder to test.
    """
    ticket = get_ticket(session, ticket_id)
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket {ticket_id!r} not found",
        )
    return TicketDetail(
        ticket_id=ticket.ticket_id,
        customer_id=ticket.customer_id,
        customer_plan=ticket.customer_plan,
        text=ticket.text or "",
        created_at=ticket.created_at.isoformat() if ticket.created_at else "",
        predictions=[_prediction_to_response(p) for p in ticket.predictions],
    )


__all__ = ["router"]
