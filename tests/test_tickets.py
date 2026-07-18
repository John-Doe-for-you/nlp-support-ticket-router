"""Tests for the Day 16 ``/tickets`` history endpoints.

Coverage map (mirrors ``docs/PROJECT_PLAN.md`` Day 16):

* ``GET /tickets`` returns a paginated, newest-first list with
  ``count``/``total``/``limit``/``offset`` populated, each item
  embedding its latest prediction and a truncated ``text_preview``.
* Empty table returns ``items: []`` and ``total: 0`` - never 404.
* Pagination honors ``limit`` + ``offset`` query params and clamps
  out-of-range values.
* ``text_preview`` is truncated with an ellipsis when the raw text
  exceeds the cap, and matches the text exactly when it does not.
* ``GET /tickets/{ticket_id}`` returns the full detail (text + every
  prediction in newest-first order) for a known id, and 404 for an
  unknown id.
* Re-runs of the same ``ticket_id`` (Day 13 upsert behavior) are
  preserved as additional ``Prediction`` rows in the detail view,
  but the list view keeps just the latest.

Like the Day 15 classify tests, we use a private in-memory SQLite
engine and seed it directly via the repository so the tests are
fast, deterministic, and don't need the trained artifact.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from ticket_router.api import main as api_main
from ticket_router.config import settings
from ticket_router.db import get_engine, get_session_factory, save_classification
from ticket_router.db.database import get_db
from ticket_router.db.models import Base
from ticket_router.pipeline.inference import (
    PredictionResult,
    reset_default_pipeline,
)
from ticket_router.schemas import MAX_TICKETS_LIMIT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    *,
    ticket_id: str,
    category: str = "Billing",
    sentiment: str = "Angry",
    priority: str = "P1",
    priority_score: int = 80,
    routed_to: str = "billing-team",
    text: str = "I was charged twice for my subscription",
    customer_id: str | None = "cus_seed",
    customer_plan: str = "pro",
) -> PredictionResult:
    """Build a :class:`PredictionResult` for seeding the in-memory DB."""
    return PredictionResult(
        ticket_id=ticket_id,
        text=text,
        customer_plan=customer_plan,
        customer_id=customer_id,
        category=category,
        category_confidence=0.9,
        sentiment=sentiment,
        sentiment_scores={"neg": 0.7, "neu": 0.2, "pos": 0.1, "compound": -0.5},
        priority=priority,
        priority_score=priority_score,
        priority_breakdown={
            "urgency_keyword_matches": 1.0,
            "negative_sentiment_intensity": 0.9,
            "customer_plan_weight": 0.5,
            "category_confidence": 0.9,
        },
        routed_to=routed_to,
        urgency_signals=["charged twice"],
        latency_ms=20,
    )


@pytest.fixture()
def in_memory_db():
    """Build a fully isolated in-memory SQLite engine + factory."""
    eng = get_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    factory = get_session_factory("sqlite:///:memory:")
    factory.configure(bind=eng)
    return eng, factory


@pytest.fixture()
def client_with_in_memory_db(in_memory_db, monkeypatch: pytest.MonkeyPatch):
    """Build a FastAPI :class:`TestClient` backed by an in-memory DB.

    We do **not** load the category model here - the /tickets routes
    are read-only and don't touch the pipeline. We do still need to
    point ``settings.database_url`` somewhere so the lifespan doesn't
    touch the real on-disk DB; the lifespan's ``init_db`` is
    overridden to a no-op.
    """
    eng, factory = in_memory_db

    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    monkeypatch.setattr(api_main, "init_db", lambda url=None: None)

    def _override_get_db():
        s = factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    app = api_main.create_app()
    app.dependency_overrides[get_db] = _override_get_db

    reset_default_pipeline()
    with TestClient(app) as client:
        yield client, eng, factory


def _seed(eng, results: list[PredictionResult]) -> None:
    """Insert each :class:`PredictionResult` into the in-memory engine."""
    Session_ = sessionmaker(bind=eng, expire_on_commit=False, autoflush=False)
    with Session_() as s:
        for r in results:
            save_classification(s, r, commit=False)
        s.commit()


# ---------------------------------------------------------------------------
# GET /tickets - empty
# ---------------------------------------------------------------------------


class TestListTicketsEmpty:
    def test_empty_table_returns_empty_page(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.get("/tickets")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == {
            "count": 0,
            "total": 0,
            "limit": 50,
            "offset": 0,
            "items": [],
        }


# ---------------------------------------------------------------------------
# GET /tickets - happy path
# ---------------------------------------------------------------------------


class TestListTicketsHappy:
    def test_returns_each_seeded_ticket_with_latest_prediction(
        self, client_with_in_memory_db
    ):
        client, eng, _ = client_with_in_memory_db
        _seed(
            eng,
            [
                _make_result(ticket_id="tkt_aaaa1111", category="Billing", routed_to="billing-team"),
                _make_result(ticket_id="tkt_bbbb2222", category="Authentication", routed_to="identity-team", priority="P2"),
                _make_result(ticket_id="tkt_cccc3333", category="Bug Report", routed_to="engineering-team", priority="P3"),
            ],
        )

        r = client.get("/tickets")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 3
        assert body["total"] == 3
        assert body["limit"] == 50
        assert body["offset"] == 0
        assert len(body["items"]) == 3

        # Every item has a ticket_id, a preview, and a latest_prediction.
        for item in body["items"]:
            assert item["ticket_id"].startswith("tkt_")
            assert "text_preview" in item
            assert "text_length" in item
            assert item["latest_prediction"] is not None
            assert item["latest_prediction"]["category"] in {
                "Billing",
                "Authentication",
                "Bug Report",
                "Feature Request",
                "Technical Setup",
            }

    def test_text_preview_truncates_long_text_with_ellipsis(
        self, client_with_in_memory_db
    ):
        client, eng, _ = client_with_in_memory_db
        long_text = "x" * 500
        _seed(eng, [_make_result(ticket_id="tkt_long0001", text=long_text)])
        body = client.get("/tickets").json()
        item = body["items"][0]
        assert item["text_length"] == 500
        # 160-char cap (see tickets._TEXT_PREVIEW_LIMIT), so 159 x's + ellipsis
        assert item["text_preview"].endswith("\u2026")
        assert len(item["text_preview"]) == 160

    def test_text_preview_keeps_short_text_intact(self, client_with_in_memory_db):
        client, eng, _ = client_with_in_memory_db
        _seed(eng, [_make_result(ticket_id="tkt_short001", text="short text")])
        body = client.get("/tickets").json()
        item = body["items"][0]
        assert item["text_preview"] == "short text"
        assert item["text_length"] == len("short text")

    def test_customer_id_optional_is_propagated(self, client_with_in_memory_db):
        client, eng, _ = client_with_in_memory_db
        _seed(
            eng,
            [
                _make_result(ticket_id="tkt_withcus1", customer_id="cus_xyz"),
                _make_result(ticket_id="tkt_nocus001", customer_id=None),
            ],
        )
        body = client.get("/tickets").json()
        by_id = {item["ticket_id"]: item for item in body["items"]}
        assert by_id["tkt_withcus1"]["customer_id"] == "cus_xyz"
        assert by_id["tkt_nocus001"]["customer_id"] is None


# ---------------------------------------------------------------------------
# GET /tickets - pagination
# ---------------------------------------------------------------------------


class TestListTicketsPagination:
    def test_limit_and_offset_applied(self, client_with_in_memory_db):
        client, eng, _ = client_with_in_memory_db
        ids = [f"tkt_pg{i:07d}" for i in range(10)]
        _seed(eng, [_make_result(ticket_id=tid) for tid in ids])

        r = client.get("/tickets", params={"limit": 3, "offset": 2})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 3
        assert body["total"] == 10
        assert body["limit"] == 3
        assert body["offset"] == 2
        assert len(body["items"]) == 3
        # ``list_tickets`` orders by created_at DESC, ticket_id DESC;
        # the 10 ids are inserted in ascending order with effectively
        # identical created_at, so the secondary key (ticket_id DESC)
        # determines the newest-first order: 9, 8, 7, ..., 0.
        # offset=2, limit=3 -> positions 2, 3, 4 -> 7, 6, 5.
        returned_ids = [item["ticket_id"] for item in body["items"]]
        assert returned_ids == [ids[7], ids[6], ids[5]]

    def test_offset_beyond_total_returns_empty_page(self, client_with_in_memory_db):
        client, eng, _ = client_with_in_memory_db
        _seed(eng, [_make_result(ticket_id=f"tkt_off{i:04d}") for i in range(2)])
        r = client.get("/tickets", params={"offset": 100})
        body = r.json()
        assert body["count"] == 0
        assert body["total"] == 2
        assert body["items"] == []

    def test_limit_clamped_to_max(self, client_with_in_memory_db):
        """``limit`` above MAX_TICKETS_LIMIT is rejected with 422 by FastAPI."""
        client, _, _ = client_with_in_memory_db
        r = client.get("/tickets", params={"limit": MAX_TICKETS_LIMIT + 1})
        assert r.status_code == 422

    def test_limit_below_one_returns_422(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.get("/tickets", params={"limit": 0})
        assert r.status_code == 422

    def test_negative_offset_returns_422(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.get("/tickets", params={"offset": -1})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /tickets - re-runs
# ---------------------------------------------------------------------------


class TestListTicketsReruns:
    def test_list_view_shows_only_latest_prediction(
        self, client_with_in_memory_db
    ):
        """Re-runs of the same ticket_id add Prediction rows, but the list
        view should still show only the latest one per ticket."""
        client, eng, _ = client_with_in_memory_db
        Session_ = sessionmaker(bind=eng, expire_on_commit=False, autoflush=False)
        with Session_() as s:
            save_classification(
                s,
                _make_result(
                    ticket_id="tkt_rerun001",
                    priority="P3",
                    priority_score=30,
                    category="Bug Report",
                    routed_to="engineering-team",
                ),
                commit=False,
            )
            save_classification(
                s,
                _make_result(
                    ticket_id="tkt_rerun001",
                    priority="P1",
                    priority_score=90,
                    category="Bug Report",
                    routed_to="engineering-team",
                ),
                commit=True,
            )

        body = client.get("/tickets").json()
        assert body["count"] == 1
        item = body["items"][0]
        assert item["ticket_id"] == "tkt_rerun001"
        # The latest prediction is the second one (P1 / score 90).
        assert item["latest_prediction"]["priority"] == "P1"
        assert item["latest_prediction"]["priority_score"] == 90


# ---------------------------------------------------------------------------
# GET /tickets/{ticket_id}
# ---------------------------------------------------------------------------


class TestGetTicket:
    def test_returns_full_detail_with_every_prediction(
        self, client_with_in_memory_db
    ):
        client, eng, _ = client_with_in_memory_db
        Session_ = sessionmaker(bind=eng, expire_on_commit=False, autoflush=False)
        with Session_() as s:
            save_classification(
                s,
                _make_result(
                    ticket_id="tkt_detail01",
                    priority="P3",
                    priority_score=20,
                    sentiment="Neutral",
                ),
                commit=False,
            )
            save_classification(
                s,
                _make_result(
                    ticket_id="tkt_detail01",
                    priority="P1",
                    priority_score=85,
                    sentiment="Angry",
                ),
                commit=True,
            )

        r = client.get("/tickets/tkt_detail01")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ticket_id"] == "tkt_detail01"
        assert body["customer_id"] == "cus_seed"
        assert body["customer_plan"] == "pro"
        assert body["text"].startswith("I was charged twice")
        assert isinstance(body["created_at"], str) and body["created_at"]
        # Both predictions are present, newest first.
        assert len(body["predictions"]) == 2
        assert body["predictions"][0]["priority"] == "P1"
        assert body["predictions"][1]["priority"] == "P3"
        # Each prediction has the full public schema
        for pred in body["predictions"]:
            for k in (
                "id",
                "category",
                "category_confidence",
                "sentiment",
                "sentiment_scores",
                "priority",
                "priority_score",
                "routed_to",
                "urgency_signals",
                "latency_ms",
                "created_at",
            ):
                assert k in pred, f"missing {k}"
            for sk in ("neg", "neu", "pos", "compound"):
                assert sk in pred["sentiment_scores"]

    def test_unknown_id_returns_404(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.get("/tickets/tkt_doesnotex")
        assert r.status_code == 404
        body = r.json()
        assert "detail" in body
        assert "tkt_doesnotex" in body["detail"]

    def test_empty_table_returns_404(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.get("/tickets/tkt_anything")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Mount points
# ---------------------------------------------------------------------------


def test_tickets_routes_are_mounted_on_app() -> None:
    """The Day 16 contract is that /tickets* appear in the OpenAPI schema."""
    from fastapi import APIRouter
    from fastapi.routing import APIRoute

    from ticket_router.api.routes import tickets as tickets_routes

    app = api_main.create_app()

    def _collect(routes) -> list[str]:
        out: list[str] = []
        for r in routes:
            if isinstance(r, APIRoute):
                out.append(r.path)
            elif isinstance(r, APIRouter):
                out.extend(_collect(r.routes))
            else:
                inner = getattr(r, "original_router", None) or getattr(r, "router", None)
                if isinstance(inner, APIRouter):
                    out.extend(_collect(inner.routes))
        return out

    paths = _collect(app.routes)
    assert "/tickets" in paths
    assert "/tickets/{ticket_id}" in paths
    assert isinstance(tickets_routes.router, APIRouter)
