"""Tests for the Day 16 ``/stats`` aggregation endpoints.

Coverage map (mirrors ``docs/PROJECT_PLAN.md`` Day 16):

* ``GET /stats`` returns four facet breakdowns plus table-wide
  ``total_predictions`` and ``total_tickets``, even on an empty DB.
* Each facet is sorted by descending count (the Day 13
  ``count_by_column`` ordering is preserved through the API).
* Counts are correct after seeding a small mixed dataset.
* ``GET /stats/{facet}`` returns a single facet, with the same
  shape as one entry of the bulk response.
* ``GET /stats/{facet}`` returns 400 for an unknown facet name
  (not 404 - the URL is well-formed, the *value* is bad).
* Re-runs of the same ticket are counted per-prediction (not
  per-ticket), matching ``count_predictions`` semantics.

Like ``test_tickets.py`` we use a private in-memory SQLite engine
and seed directly via the repository so the tests stay fast and
deterministic.
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
    customer_id: str | None = "cus_seed",
) -> PredictionResult:
    return PredictionResult(
        ticket_id=ticket_id,
        text="seed",
        customer_plan="pro",
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
        urgency_signals=["x"],
        latency_ms=20,
    )


@pytest.fixture()
def in_memory_db():
    eng = get_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    factory = get_session_factory("sqlite:///:memory:")
    factory.configure(bind=eng)
    return eng, factory


@pytest.fixture()
def client_with_in_memory_db(in_memory_db, monkeypatch: pytest.MonkeyPatch):
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
    Session_ = sessionmaker(bind=eng, expire_on_commit=False, autoflush=False)
    with Session_() as s:
        for r in results:
            save_classification(s, r, commit=False)
        s.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _breakdown_to_dict(breakdown: dict) -> dict[str, int]:
    """Flatten a ``StatsBreakdown`` JSON dict into ``{label: count}``."""
    out: dict[str, int] = {}
    for entry in breakdown["items"]:
        out.update(entry)
    return out


# ---------------------------------------------------------------------------
# GET /stats - empty
# ---------------------------------------------------------------------------


class TestStatsEmpty:
    def test_empty_db_returns_zero_everywhere(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.get("/stats")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total_predictions"] == 0
        assert body["total_tickets"] == 0
        for facet in ("by_category", "by_sentiment", "by_priority", "by_team"):
            assert body[facet]["total"] == 0
            assert body[facet]["items"] == []


# ---------------------------------------------------------------------------
# GET /stats - happy path
# ---------------------------------------------------------------------------


class TestStatsHappy:
    def test_counts_match_seeded_data(self, client_with_in_memory_db):
        client, eng, _ = client_with_in_memory_db
        _seed(
            eng,
            [
                _make_result(ticket_id="tkt_s0000001", category="Billing", sentiment="Angry", priority="P1", routed_to="billing-team"),
                _make_result(ticket_id="tkt_s0000002", category="Billing", sentiment="Frustrated", priority="P2", routed_to="billing-team"),
                _make_result(ticket_id="tkt_s0000003", category="Authentication", sentiment="Neutral", priority="P3", routed_to="identity-team"),
                _make_result(ticket_id="tkt_s0000004", category="Bug Report", sentiment="Angry", priority="P1", routed_to="engineering-team"),
            ],
        )

        r = client.get("/stats")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total_predictions"] == 4
        assert body["total_tickets"] == 4

        by_cat = _breakdown_to_dict(body["by_category"])
        assert by_cat == {"Billing": 2, "Authentication": 1, "Bug Report": 1}

        by_sent = _breakdown_to_dict(body["by_sentiment"])
        assert by_sent == {"Angry": 2, "Frustrated": 1, "Neutral": 1}

        by_pri = _breakdown_to_dict(body["by_priority"])
        assert by_pri == {"P1": 2, "P2": 1, "P3": 1}

        by_team = _breakdown_to_dict(body["by_team"])
        assert by_team == {
            "billing-team": 2,
            "identity-team": 1,
            "engineering-team": 1,
        }

    def test_breakdowns_sorted_by_descending_count(
        self, client_with_in_memory_db
    ):
        client, eng, _ = client_with_in_memory_db
        # 3x Billing, 1x Auth, 2x Bug Report -> Billing first, then Bug Report, then Auth
        _seed(
            eng,
            [
                _make_result(ticket_id="tkt_o0000001", category="Bug Report", routed_to="engineering-team"),
                _make_result(ticket_id="tkt_o0000002", category="Billing", routed_to="billing-team"),
                _make_result(ticket_id="tkt_o0000003", category="Billing", routed_to="billing-team"),
                _make_result(ticket_id="tkt_o0000004", category="Authentication", routed_to="identity-team"),
                _make_result(ticket_id="tkt_o0000005", category="Billing", routed_to="billing-team"),
                _make_result(ticket_id="tkt_o0000006", category="Bug Report", routed_to="engineering-team"),
            ],
        )

        body = client.get("/stats").json()
        cat_keys = [list(item.keys())[0] for item in body["by_category"]["items"]]
        assert cat_keys == ["Billing", "Bug Report", "Authentication"]
        # And the totals line up.
        assert body["by_category"]["total"] == 6

    def test_reruns_count_per_prediction(self, client_with_in_memory_db):
        """Re-running the same ticket_id adds Prediction rows, so
        ``total_predictions`` exceeds ``total_tickets``. The facet
        counts reflect predictions, not tickets."""
        client, eng, _ = client_with_in_memory_db
        Session_ = sessionmaker(bind=eng, expire_on_commit=False, autoflush=False)
        with Session_() as s:
            save_classification(
                s,
                _make_result(
                    ticket_id="tkt_r0000001",
                    priority="P3",
                    priority_score=20,
                ),
                commit=False,
            )
            save_classification(
                s,
                _make_result(
                    ticket_id="tkt_r0000001",
                    priority="P1",
                    priority_score=90,
                ),
                commit=True,
            )

        body = client.get("/stats").json()
        assert body["total_tickets"] == 1
        assert body["total_predictions"] == 2
        by_pri = _breakdown_to_dict(body["by_priority"])
        assert by_pri == {"P1": 1, "P3": 1}


# ---------------------------------------------------------------------------
# GET /stats/{facet}
# ---------------------------------------------------------------------------


class TestStatsFacet:
    @pytest.mark.parametrize(
        "facet,expected",
        [
            ("category", {"Billing": 2, "Authentication": 1, "Bug Report": 1}),
            ("sentiment", {"Angry": 2, "Frustrated": 1, "Neutral": 1}),
            ("priority", {"P1": 2, "P2": 1, "P3": 1}),
            ("team", {"billing-team": 2, "identity-team": 1, "engineering-team": 1}),
        ],
    )
    def test_facet_returns_correct_counts(
        self, client_with_in_memory_db, facet, expected
    ):
        client, eng, _ = client_with_in_memory_db
        _seed(
            eng,
            [
                _make_result(ticket_id="tkt_f0000001", category="Billing", sentiment="Angry", priority="P1", routed_to="billing-team"),
                _make_result(ticket_id="tkt_f0000002", category="Billing", sentiment="Frustrated", priority="P2", routed_to="billing-team"),
                _make_result(ticket_id="tkt_f0000003", category="Authentication", sentiment="Neutral", priority="P3", routed_to="identity-team"),
                _make_result(ticket_id="tkt_f0000004", category="Bug Report", sentiment="Angry", priority="P1", routed_to="engineering-team"),
            ],
        )

        r = client.get(f"/stats/{facet}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["facet"] == facet
        assert body["total"] == sum(expected.values())
        flat = _breakdown_to_dict({"items": body["items"]})
        assert flat == expected

    def test_unknown_facet_returns_400(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.get("/stats/bogus")
        assert r.status_code == 400
        body = r.json()
        assert "detail" in body
        assert "bogus" in body["detail"]

    def test_facet_on_empty_db(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.get("/stats/category")
        assert r.status_code == 200
        body = r.json()
        assert body == {"facet": "category", "total": 0, "items": []}


# ---------------------------------------------------------------------------
# Mount points
# ---------------------------------------------------------------------------


def test_stats_routes_are_mounted_on_app() -> None:
    from fastapi import APIRouter
    from fastapi.routing import APIRoute

    from ticket_router.api.routes import stats as stats_routes

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
    assert "/stats" in paths
    assert "/stats/{facet}" in paths
    assert isinstance(stats_routes.router, APIRouter)
