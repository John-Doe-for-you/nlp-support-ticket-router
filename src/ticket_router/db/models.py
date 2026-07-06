"""SQLAlchemy ORM models for the support ticket router.

We model the world as two related tables:

* ``tickets``  : one row per incoming support ticket. Holds the raw text,
                 customer identifiers, and timestamps. The natural key is
                 ``ticket_id`` (the ``tkt_xxxxxxxx`` string minted by the
                 Day 11 inference pipeline). It is also the primary key,
                 so duplicates are impossible by construction.
* ``predictions``: one row per classification result, joined to ``tickets``
                 on ``ticket_id``. We keep prediction fields denormalized
                 (string label + numeric confidence) rather than linking
                 out to a ``categories`` table, because:

                 1) The label set is locked in ``docs/PROJECT_PLAN.md`` §5
                    and never changes at runtime.
                 2) It keeps Day 15's API serialization a straight row
                    read, no joins.
                 3) It preserves the full prediction as it was returned
                    even if we later retrain the model with new labels
                    (history is auditable).

One ticket has a 1:1 relationship with one prediction in our flow
(the pipeline always returns exactly one result), but the schema is
written so that re-runs create additional prediction rows (a useful
debugging trail). The Day 15 API will fetch the *latest* prediction
when serving history.

All timestamps are stored as naive UTC ``DateTime`` to keep SQLite
happy (``TIMESTAMP`` semantics differ across dialects). The repository
is responsible for stamping them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models in the project."""


class Ticket(Base):
    """An incoming support ticket.

    Columns
    -------
    ticket_id
        Application-generated id of the form ``tkt_xxxxxxxx`` (see
        ``pipeline.inference._new_ticket_id``). Used as the primary key
        so the API can do point lookups without an extra index.
    text
        Raw ticket text exactly as the customer submitted it. Stored
        as ``Text`` so we don't truncate long pastes.
    customer_id
        Optional opaque customer identifier (e.g. ``"cus_123"``).
    customer_plan
        One of ``"free"``, ``"pro"``, ``"enterprise"``. Not normalized
        to an enum table - same reasoning as the prediction labels
        above (locked set, small, no joins needed).
    created_at
        UTC timestamp the row was inserted. Defaulted to ``now()`` at
        the database level so the repository doesn't have to set it.
    """

    __tablename__ = "tickets"

    ticket_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    customer_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    customer_plan: Mapped[str] = mapped_column(String(16), nullable=False, default="free")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction",
        back_populates="ticket",
        cascade="all, delete-orphan",
        order_by="Prediction.created_at.desc()",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Ticket ticket_id={self.ticket_id!r} "
            f"customer_id={self.customer_id!r} plan={self.customer_plan!r}>"
        )


class Prediction(Base):
    """A single classification result produced for a ticket.

    The composite index on ``(ticket_id, created_at DESC)`` makes the
    "latest prediction for this ticket" query (the Day 16 history
    endpoint) a single index seek.
    """

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_ticket_created", "ticket_id", "created_at"),
        Index("ix_predictions_category", "category"),
        Index("ix_predictions_priority", "priority"),
        Index("ix_predictions_routed_to", "routed_to"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticket_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("tickets.ticket_id", ondelete="CASCADE"),
        nullable=False,
    )

    category: Mapped[str] = mapped_column(String(32), nullable=False)
    category_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment: Mapped[str] = mapped_column(String(16), nullable=False)
    sentiment_neg: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_neu: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_pos: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_compound: Mapped[float] = mapped_column(Float, nullable=False)
    priority: Mapped[str] = mapped_column(String(4), nullable=False)
    priority_score: Mapped[int] = mapped_column(Integer, nullable=False)
    routed_to: Mapped[str] = mapped_column(String(64), nullable=False)
    urgency_signals: Mapped[str] = mapped_column(Text, nullable=False, default="")
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    ticket: Mapped["Ticket"] = relationship("Ticket", back_populates="predictions")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Prediction id={self.id} ticket_id={self.ticket_id!r} "
            f"category={self.category!r} priority={self.priority!r}>"
        )


__all__ = ["Base", "Ticket", "Prediction"]
