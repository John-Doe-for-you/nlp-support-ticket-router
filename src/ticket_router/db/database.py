"""SQLAlchemy engine, session factory, and schema bootstrap.

Public API
----------
* ``get_engine(url)``       : build (and cache) a :class:`Engine` for a
                              given database URL. Defaults to the URL
                              in :data:`settings.database_url`.
* ``get_session_factory()`` : return a process-wide
                              :class:`sessionmaker` bound to the default
                              engine. Bind a *different* factory with
                              ``get_session_factory(url=...)`` in tests
                              to get an isolated in-memory DB.
* ``get_db()``              : FastAPI dependency yielding a request-scoped
                              :class:`Session` and committing/rolling back
                              on the way out.
* ``init_db()``             : create all tables. Safe to call repeatedly;
                              no-op if everything already exists.
* ``reset_db()``            : drop + recreate all tables. Test-only helper.

Design notes
------------
We use the static pool for SQLite-in-memory tests so every connection
sees the same schema (regular pools would each open a private DB).
The "real" engine (file-backed SQLite) uses the default pool. The
``check_same_thread`` flag is required for SQLite + multi-threaded
servers like Uvicorn.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Optional

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ticket_router.config import settings
from ticket_router.db.models import Base

_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[sessionmaker[Session]] = None


def _build_engine(url: str) -> Engine:
    """Construct an :class:`Engine` appropriate for the given URL.

    SQLite (including the in-memory variant) gets the small set of
    event listeners and pool tweaks needed to behave well under
    multi-threaded use. Other dialects get a vanilla engine.
    """
    parsed = make_url(url)
    connect_args: dict[str, object] = {}
    engine_kwargs: dict[str, object] = {}

    if parsed.drivername.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # In-memory SQLite: share a single connection across "sessions"
        # so the schema created by one thread is visible to another.
        if parsed.database in (None, "", ":memory:"):
            engine_kwargs["poolclass"] = StaticPool

    engine = create_engine(url, connect_args=connect_args, future=True, **engine_kwargs)

    if parsed.drivername.startswith("sqlite"):
        # Enforce FK constraints on SQLite (off by default for legacy reasons).
        @event.listens_for(engine, "connect")
        def _enable_sqlite_fk(dbapi_conn, _connection_record):  # pragma: no cover
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def get_engine(url: Optional[str] = None) -> Engine:
    """Return the process-wide :class:`Engine`, building it on first use.

    Pass an explicit ``url`` to build a one-off engine (e.g. inside a
    test that wants an isolated DB). The result is *not* cached when
    ``url`` is provided, so the caller owns its lifecycle.
    """
    global _ENGINE
    if url is None:
        if _ENGINE is None:
            _ENGINE = _build_engine(settings.database_url)
        return _ENGINE
    return _build_engine(url)


def get_session_factory(url: Optional[str] = None) -> sessionmaker[Session]:
    """Return a :class:`sessionmaker` bound to the chosen engine.

    With no arguments, returns the cached process-wide factory. Pass
    ``url`` to build a one-off factory for a test database.
    """
    global _SESSION_FACTORY
    if url is None:
        if _SESSION_FACTORY is None:
            _SESSION_FACTORY = sessionmaker(
                bind=get_engine(), expire_on_commit=False, autoflush=False
            )
        return _SESSION_FACTORY
    return sessionmaker(bind=_build_engine(url), expire_on_commit=False, autoflush=False)


def init_db(url: Optional[str] = None) -> None:
    """Create all tables that don't already exist. Idempotent."""
    engine = get_engine(url)
    Base.metadata.create_all(engine)


def reset_db(url: Optional[str] = None) -> None:
    """Drop and recreate every table. Intended for tests only."""
    engine = get_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a :class:`Session` per request.

    Commits on clean exit, rolls back on exception. The session is
    always closed, even on error, so SQLite file locks are released
    promptly.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "get_engine",
    "get_session_factory",
    "get_db",
    "init_db",
    "reset_db",
]
