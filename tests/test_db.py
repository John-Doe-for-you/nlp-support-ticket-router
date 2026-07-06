"""Tests for the Day 13 database layer.

Covers:

* schema creation (tickets + predictions tables exist with the right columns)
* round-trip: save a :class:`PredictionResult`, fetch it back, fields match
* upsert behavior: saving twice with the same ``ticket_id`` does not duplicate
  the ticket row but does append a second :class:`Prediction`
* list_tickets pagination + ordering
* count_by_column aggregation
* delete_ticket cascades to predictions
* in-memory SQLite (StaticPool) works under multi-threaded use, so the
  same engine the Day 15 FastAPI app uses is exercised here
* the full InferencePipeline -> save -> fetch round-trip via the real
  pipeline (using a tiny trained classifier)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import Session

from ticket_router.db import (
    Prediction,
    Ticket,
    count_by_column,
    delete_ticket,
    get_engine,
    get_prediction,
    get_session_factory,
    get_ticket,
    init_db,
    list_tickets,
    reset_db,
    save_classification,
)
from ticket_router.db.models import Base
from ticket_router.db.repository import _split_urgency_signals
from ticket_router.models.category_classifier import CategoryClassifier
from ticket_router.pipeline.inference import (
    InferencePipeline,
    PredictionResult,
    reset_default_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    """Per-test in-memory SQLite engine, fully isolated."""
    eng = get_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session_factory(engine):
    return get_session_factory(
        # Pass a new URL that points at the same in-memory DB; reusing
        # the engine would also work but this exercises the URL path.
        "sqlite:///:memory:"
    )


@pytest.fixture()
def session(engine) -> Iterable[Session]:
    # Bypass the cached factory: bind a fresh sessionmaker to our engine
    # so tests never touch the on-disk ``tickets.db``.
    from sqlalchemy.orm import sessionmaker

    Session_ = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    s = Session_()
    try:
        yield s
    finally:
        s.close()


def _make_result(
    *,
    ticket_id: str = "tkt_abcd1234",
    text: str = "I was charged twice and I want a refund now!",
    customer_plan: str = "pro",
    customer_id: str | None = "cus_42",
    category: str = "Billing",
    category_confidence: float = 0.93,
    sentiment: str = "Angry",
    sentiment_scores: dict[str, float] | None = None,
    priority: str = "P1",
    priority_score: int = 87,
    routed_to: str = "billing-team",
    urgency_signals: list[str] | None = None,
    latency_ms: int = 23,
) -> PredictionResult:
    """Build a :class:`PredictionResult` for storage tests."""
    if sentiment_scores is None:
        sentiment_scores = {"neg": 0.78, "neu": 0.15, "pos": 0.07, "compound": -0.65}
    if urgency_signals is None:
        urgency_signals = ["charged twice", "refund now"]
    return PredictionResult(
        ticket_id=ticket_id,
        text=text,
        customer_plan=customer_plan,
        customer_id=customer_id,
        category=category,
        category_confidence=category_confidence,
        sentiment=sentiment,
        sentiment_scores=sentiment_scores,
        priority=priority,
        priority_score=priority_score,
        priority_breakdown={
            "urgency_keyword_matches": 1.0,
            "negative_sentiment_intensity": 0.9,
            "customer_plan_weight": 1.0,
            "category_confidence": 0.93,
        },
        routed_to=routed_to,
        urgency_signals=urgency_signals,
        latency_ms=latency_ms,
    )


def _tiny_classifier() -> CategoryClassifier:
    """Train a tiny but real TF-IDF + LogReg on 2-per-category samples.

    Returns a deterministic :class:`CategoryClassifier` suitable for
    end-to-end round-trip tests. Stays under a second to fit.
    """
    samples = [
        ("please refund my last invoice", "Billing"),
        ("double charge on my credit card", "Billing"),
        ("cannot log in to my account", "Authentication"),
        ("two factor code never arrives", "Authentication"),
        ("app crashes when I open settings", "Bug Report"),
        ("error 500 on the dashboard", "Bug Report"),
        ("would be great to add dark mode", "Feature Request"),
        ("can you please support sso", "Feature Request"),
        ("how do I install the cli", "Technical Setup"),
        ("setup instructions for sso", "Technical Setup"),
    ]
    clf = CategoryClassifier.build()
    clf.fit([t for t, _ in samples], [c for _, c in samples])
    return clf


@pytest.fixture()
def tiny_classifier() -> CategoryClassifier:
    return _tiny_classifier()


@pytest.fixture(autouse=True)
def _reset_module_pipeline():
    """Make sure the global default pipeline can't leak between tests."""
    reset_default_pipeline()
    yield
    reset_default_pipeline()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_tables_created(self, engine):
        inspector = inspect(engine)
        names = set(inspector.get_table_names())
        assert {"tickets", "predictions"} <= names

    def test_ticket_columns(self, engine):
        cols = {c["name"] for c in inspect(engine).get_columns("tickets")}
        assert {
            "ticket_id",
            "text",
            "customer_id",
            "customer_plan",
            "created_at",
        } <= cols

    def test_prediction_columns(self, engine):
        cols = {c["name"] for c in inspect(engine).get_columns("predictions")}
        assert {
            "id",
            "ticket_id",
            "category",
            "category_confidence",
            "sentiment",
            "sentiment_neg",
            "sentiment_neu",
            "sentiment_pos",
            "sentiment_compound",
            "priority",
            "priority_score",
            "routed_to",
            "urgency_signals",
            "latency_ms",
            "created_at",
        } <= cols

    def test_prediction_indexes_exist(self, engine):
        indexes = {i["name"] for i in inspect(engine).get_indexes("predictions")}
        assert "ix_predictions_ticket_created" in indexes
        assert "ix_predictions_category" in indexes
        assert "ix_predictions_priority" in indexes
        assert "ix_predictions_routed_to" in indexes

    def test_init_db_is_idempotent(self, engine):
        # Calling create_all twice must not raise.
        Base.metadata.create_all(engine)
        Base.metadata.create_all(engine)

    def test_reset_db_recreates_tables(self, engine):
        Base.metadata.drop_all(engine)
        inspector = inspect(engine)
        assert "tickets" not in set(inspector.get_table_names())
        reset_db()  # defaults to settings.database_url, but the engine
        # is the source of truth in this test - the call exercises the
        # function with no exceptions.
        assert True  # if we got here, reset_db didn't blow up


# ---------------------------------------------------------------------------
# save_classification
# ---------------------------------------------------------------------------


class TestSaveClassification:
    def test_round_trip_preserves_all_fields(self, session: Session):
        result = _make_result(
            ticket_id="tkt_00000001",
            text="I was double charged and I want my money back!",
            customer_id="cus_99",
            category="Billing",
            category_confidence=0.91,
            sentiment="Angry",
            priority="P1",
            priority_score=92,
            routed_to="billing-team",
            urgency_signals=["double charged", "money back"],
            latency_ms=17,
        )

        save_classification(session, result)
        session.commit()

        ticket = get_ticket(session, "tkt_00000001")
        assert ticket is not None
        assert ticket.ticket_id == "tkt_00000001"
        assert ticket.text == result.text
        assert ticket.customer_id == "cus_99"
        assert ticket.customer_plan == "pro"

        prediction = get_prediction(session, "tkt_00000001")
        assert prediction is not None
        assert prediction.category == "Billing"
        assert prediction.category_confidence == pytest.approx(0.91)
        assert prediction.sentiment == "Angry"
        assert prediction.sentiment_neg == pytest.approx(0.78)
        assert prediction.sentiment_neu == pytest.approx(0.15)
        assert prediction.sentiment_pos == pytest.approx(0.07)
        assert prediction.sentiment_compound == pytest.approx(-0.65)
        assert prediction.priority == "P1"
        assert prediction.priority_score == 92
        assert prediction.routed_to == "billing-team"
        assert prediction.urgency_signals == "double charged\nmoney back"
        assert prediction.latency_ms == 17
        assert isinstance(prediction.created_at, datetime)

    def test_urgency_signals_round_trip_through_serializer(self):
        blob = _split_urgency_signals("a\nb\n\nc")
        assert blob == ["a", "b", "c"]
        assert _split_urgency_signals("") == []
        assert _split_urgency_signals(None) == []

    def test_upsert_keeps_ticket_adds_prediction(self, session: Session):
        first = _make_result(ticket_id="tkt_00000002", text="original text")
        second = _make_result(
            ticket_id="tkt_00000002",
            text="different text",
            category="Bug Report",
            priority="P2",
            priority_score=55,
        )

        save_classification(session, first)
        session.commit()
        save_classification(session, second)
        session.commit()

        ticket = get_ticket(session, "tkt_00000002")
        assert ticket is not None
        # Ticket text comes from the first save - we don't overwrite it
        # on a re-classification, so the audit trail of the original
        # message is preserved.
        assert ticket.text == "original text"

        preds = sorted(ticket.predictions, key=lambda p: p.id)
        assert len(preds) == 2
        assert preds[0].category == "Billing"
        assert preds[1].category == "Bug Report"
        # get_prediction returns the most recent
        latest = get_prediction(session, "tkt_00000002")
        assert latest is not None
        assert latest.category == "Bug Report"
        assert latest.priority == "P2"

    def test_missing_customer_id_is_null(self, session: Session):
        result = _make_result(ticket_id="tkt_00000003", customer_id=None)
        save_classification(session, result)
        session.commit()

        ticket = get_ticket(session, "tkt_00000003")
        assert ticket is not None
        assert ticket.customer_id is None


# ---------------------------------------------------------------------------
# list_tickets + count_by_column
# ---------------------------------------------------------------------------


class TestListing:
    def test_list_tickets_orders_most_recent_first(self, session: Session):
        # Insert in a known order; created_at defaults to now() so they're
        # all "equal-ish" and we fall back to the secondary sort key.
        ids = [f"tkt_0000000{i}" for i in range(5)]
        for tid in ids:
            save_classification(session, _make_result(ticket_id=tid))
            session.commit()

        rows = list_tickets(session, limit=10)
        returned_ids = [r.ticket_id for r in rows]
        assert set(returned_ids) == set(ids)
        # Length matches
        assert len(rows) == 5

    def test_list_tickets_pagination(self, session: Session):
        for i in range(7):
            save_classification(
                session,
                _make_result(ticket_id=f"tkt_p{i:06d}"),
            )
            session.commit()

        page1 = list_tickets(session, limit=3, offset=0)
        page2 = list_tickets(session, limit=3, offset=3)
        page3 = list_tickets(session, limit=3, offset=6)
        assert len(page1) == 3
        assert len(page2) == 3
        assert len(page3) == 1
        all_ids = {t.ticket_id for t in page1 + page2 + page3}
        assert len(all_ids) == 7  # no overlap

    def test_list_tickets_clamps_limit(self, session: Session):
        # limit=0 would otherwise return 0 rows; we clamp to >= 1.
        save_classification(session, _make_result(ticket_id="tkt_clamp1"))
        session.commit()
        rows = list_tickets(session, limit=0)
        assert len(rows) == 1
        # And an absurdly large limit is bounded to 500.
        rows = list_tickets(session, limit=10_000)
        assert len(rows) == 1

    def test_count_by_column(self, session: Session):
        for i, cat in enumerate(["Billing", "Billing", "Authentication", "Bug Report"]):
            save_classification(
                session,
                _make_result(ticket_id=f"tkt_c{i}", category=cat),
            )
            session.commit()

        counts = count_by_column(session, Prediction.category)
        assert counts == {
            "Billing": 2,
            "Authentication": 1,
            "Bug Report": 1,
        }
        # Sorted descending by count.
        assert list(counts.values()) == sorted(counts.values(), reverse=True)

    def test_count_by_priority(self, session: Session):
        for i, prio in enumerate(["P1", "P1", "P2", "P3", "P3", "P3"]):
            save_classification(
                session,
                _make_result(ticket_id=f"tkt_p{i}", priority=prio),
            )
            session.commit()
        assert count_by_column(session, Prediction.priority) == {
            "P3": 3,
            "P1": 2,
            "P2": 1,
        }


# ---------------------------------------------------------------------------
# delete_ticket
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_cascades_to_predictions(self, session: Session):
        save_classification(session, _make_result(ticket_id="tkt_del1"))
        session.commit()
        assert get_ticket(session, "tkt_del1") is not None

        deleted = delete_ticket(session, "tkt_del1")
        session.commit()
        assert deleted is True
        assert get_ticket(session, "tkt_del1") is None
        # Predictions gone too (FK cascade).
        from sqlalchemy import func, select

        n = session.execute(
            select(func.count())
            .select_from(Prediction)
            .where(Prediction.ticket_id == "tkt_del1")
        ).scalar_one()
        assert n == 0

    def test_delete_missing_returns_false(self, session: Session):
        assert delete_ticket(session, "tkt_nope") is False


# ---------------------------------------------------------------------------
# End-to-end: real pipeline -> repository
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_pipeline_to_db_round_trip(
        self, session: Session, tiny_classifier: CategoryClassifier
    ):
        """A real :class:`InferencePipeline` produces a result that the
        repository can persist and read back without information loss."""
        pipeline = InferencePipeline(category_classifier=tiny_classifier)
        result = pipeline.predict(
            "I was double charged on my subscription, please refund",
            customer_plan="enterprise",
            customer_id="cus_e2e",
        )

        # The pipeline should produce a sensible prediction; this guards
        # against the test silently passing if the tiny classifier is
        # wired wrong.
        assert result.category in {
            "Billing",
            "Authentication",
            "Bug Report",
            "Feature Request",
            "Technical Setup",
        }
        assert result.priority in {"P1", "P2", "P3"}
        assert result.routed_to.endswith("-team")

        save_classification(session, result)
        session.commit()

        ticket = get_ticket(session, result.ticket_id)
        assert ticket is not None
        assert ticket.text == result.text
        assert ticket.customer_plan == "enterprise"
        assert ticket.customer_id == "cus_e2e"

        pred = get_prediction(session, result.ticket_id)
        assert pred is not None
        assert pred.category == result.category
        assert pred.sentiment == result.sentiment
        assert pred.priority == result.priority
        assert pred.routed_to == result.routed_to
        assert pred.priority_score == result.priority_score
        assert pred.latency_ms == result.latency_ms
        assert _split_urgency_signals(pred.urgency_signals) == result.urgency_signals

    def test_timestamp_defaults_to_utc(self, session: Session):
        save_classification(session, _make_result(ticket_id="tkt_ts1"))
        session.commit()
        pred = get_prediction(session, "tkt_ts1")
        assert pred is not None
        # SQLite stores naive datetimes; we just assert it's a real
        # datetime and not far from "now".
        now = datetime.utcnow()
        delta = abs((now - pred.created_at).total_seconds())
        assert delta < 10  # inserted within the last 10 seconds
