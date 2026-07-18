"""Database layer: SQLAlchemy ORM models, engine, and repository helpers.

Day 13 deliverable per ``docs/PROJECT_PLAN.md``. Re-exports the public
API so callers can ``from ticket_router.db import ...`` without
drilling into submodules.
"""

from ticket_router.db.database import (
    get_db,
    get_engine,
    get_session_factory,
    init_db,
    reset_db,
)
from ticket_router.db.models import Base, Prediction, Ticket
from ticket_router.db.repository import (
    count_by_column,
    count_predictions,
    count_tickets,
    delete_ticket,
    get_prediction,
    get_ticket,
    list_tickets,
    save_classification,
)

__all__ = [
    "Base",
    "Prediction",
    "Ticket",
    "get_db",
    "get_engine",
    "get_session_factory",
    "init_db",
    "reset_db",
    "save_classification",
    "get_ticket",
    "get_prediction",
    "list_tickets",
    "count_by_column",
    "count_tickets",
    "count_predictions",
    "delete_ticket",
]
