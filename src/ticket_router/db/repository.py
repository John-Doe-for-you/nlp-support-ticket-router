"""CRUD repository helpers for the ticket + prediction tables.

Public API
----------
* ``save_classification(session, result, *, ticket_text, ...)``
    Persist a :class:`PredictionResult` and its parent
    :class:`Ticket` row in a single transaction. Returns the
    saved :class:`Ticket`.

* ``get_ticket(session, ticket_id)``
    Fetch a ticket by id (with its latest prediction eager-loaded).
    Returns ``None`` if not found.

* ``get_prediction(session, ticket_id)``
    Fetch the *latest* :class:`Prediction` for the given ticket id.

* ``list_tickets(session, *, limit, offset)``
    Paginated list of tickets, most recent first.

* ``count_tickets(session)``
    Total number of ticket rows. Used by the Day 16 history endpoint
    to populate ``total`` so the client can paginate.

* ``count_predictions(session)``
    Total number of prediction rows (across all tickets, not deduped).

* ``count_by_column(session, column)``
    Group-by count for any string-ish column on :class:`Prediction`
    (used by the Day 16 ``/stats`` endpoint).

* ``delete_ticket(session, ticket_id)``
    Remove a ticket and its predictions. Returns ``True`` if a row
    was deleted, ``False`` if it didn't exist.

The repository deliberately takes an explicit :class:`Session` rather
than building one internally. That keeps transactions under caller
control and makes the helpers trivially mockable in tests.
"""

from __future__ import annotations

from typing import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from ticket_router.db.models import Prediction, Ticket
from ticket_router.pipeline.inference import PredictionResult


def _join_urgency_signals(signals: Sequence[str]) -> str:
    """Serialize the urgency list as a single newline-delimited string.

    Stored as a string (not JSON) to keep SQLite happy without a
    JSON column type. The Day 15 API splits on ``"\\n"`` to rebuild
    the list - cheaper than a JSON round-trip and easy to read in
    raw ``sqlite3`` dumps.
    """
    return "\n".join(signals or [])


def _split_urgency_signals(blob: str | None) -> list[str]:
    if not blob:
        return []
    return [s for s in blob.split("\n") if s]


def save_classification(
    session: Session,
    result: PredictionResult,
    *,
    commit: bool = False,
) -> Ticket:
    """Persist a :class:`PredictionResult` together with its ticket.

    The ticket row is upserted: if one already exists with the same
    ``ticket_id`` (e.g. on a retry), we keep the original ``text``,
    ``customer_id`` and ``created_at`` and just append a new
    :class:`Prediction` row. This preserves the full audit trail.

    Parameters
    ----------
    session
        An open SQLAlchemy session. The caller controls the
        transaction boundary; pass ``commit=True`` for convenience
        in scripts.
    result
        The output of :meth:`InferencePipeline.predict`.
    """
    existing = session.get(Ticket, result.ticket_id)
    if existing is None:
        ticket = Ticket(
            ticket_id=result.ticket_id,
            text=result.text,
            customer_id=result.customer_id,
            customer_plan=result.customer_plan,
        )
        session.add(ticket)
    else:
        ticket = existing

    scores = result.sentiment_scores
    prediction = Prediction(
        ticket_id=result.ticket_id,
        category=result.category,
        category_confidence=float(result.category_confidence),
        sentiment=result.sentiment,
        sentiment_neg=float(scores.get("neg", 0.0)),
        sentiment_neu=float(scores.get("neu", 0.0)),
        sentiment_pos=float(scores.get("pos", 0.0)),
        sentiment_compound=float(scores.get("compound", 0.0)),
        priority=result.priority,
        priority_score=int(result.priority_score),
        routed_to=result.routed_to,
        urgency_signals=_join_urgency_signals(result.urgency_signals),
        latency_ms=int(result.latency_ms),
    )
    session.add(prediction)
    session.flush()  # populate prediction.id without committing
    if commit:
        session.commit()
    return ticket


def get_ticket(session: Session, ticket_id: str) -> Ticket | None:
    """Return the ticket with its predictions eager-loaded, or ``None``."""
    stmt = (
        select(Ticket)
        .where(Ticket.ticket_id == ticket_id)
        .options(selectinload(Ticket.predictions))
    )
    return session.execute(stmt).scalar_one_or_none()


def get_prediction(session: Session, ticket_id: str) -> Prediction | None:
    """Return the most recent :class:`Prediction` for ``ticket_id``."""
    stmt = (
        select(Prediction)
        .where(Prediction.ticket_id == ticket_id)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def list_tickets(
    session: Session,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[Ticket]:
    """Return tickets most-recent-first, with their latest prediction.

    ``limit`` is clamped to a sensible upper bound so a buggy caller
    can't accidentally scan the whole table.
    """
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    stmt = (
        select(Ticket)
        .options(selectinload(Ticket.predictions))
        .order_by(Ticket.created_at.desc(), Ticket.ticket_id.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(session.execute(stmt).scalars().all())


def count_by_column(session: Session, column) -> dict[str, int]:
    """Group-by count for a single :class:`Prediction` column.

    Returns a plain ``{value: count}`` dict, sorted by descending count
    so the Day 16 stats endpoint can return the most common values
    first without an extra sort.
    """
    label_col = func.coalesce(column, "<unknown>").label("value")
    stmt = select(label_col, func.count().label("n")).group_by(label_col).order_by(func.count().desc())
    return {row.value: int(row.n) for row in session.execute(stmt).all()}


def count_tickets(session: Session) -> int:
    """Return the total number of ``Ticket`` rows.

    Used by the Day 16 ``GET /tickets`` history endpoint so the
    client can paginate without an extra round-trip.
    """
    stmt = select(func.count()).select_from(Ticket)
    return int(session.execute(stmt).scalar_one())


def count_predictions(session: Session) -> int:
    """Return the total number of ``Prediction`` rows (across all tickets).

    Note this counts every prediction, including re-runs of the same
    ticket. ``count_tickets`` is the more useful "how many unique
    tickets" number; this is the "how many classifications performed"
    number.
    """
    stmt = select(func.count()).select_from(Prediction)
    return int(session.execute(stmt).scalar_one())


def delete_ticket(session: Session, ticket_id: str) -> bool:
    """Delete a ticket (cascades to its predictions). Returns success flag."""
    stmt = delete(Ticket).where(Ticket.ticket_id == ticket_id)
    result = session.execute(stmt)
    return result.rowcount > 0


__all__ = [
    "save_classification",
    "get_ticket",
    "get_prediction",
    "list_tickets",
    "count_by_column",
    "count_tickets",
    "count_predictions",
    "delete_ticket",
    "_split_urgency_signals",
]
