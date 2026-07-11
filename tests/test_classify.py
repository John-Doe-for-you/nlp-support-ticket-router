"""Tests for the Day 15 ``/classify`` and ``/classify/batch`` endpoints.

Coverage map (mirrors ``docs/PROJECT_PLAN.md`` Day 15):

* Happy-path single classification: returns a schema-valid
  :class:`ClassifyResponse` and writes one row to ``tickets`` and one
  row to ``predictions``.
* The persisted row's fields match the API response (round-trip check).
* Default values: omitting ``customer_plan`` defaults to ``"free"`` and
  omitting ``customer_id`` stores NULL.
* ``extra='forbid'`` on the request model: an unknown field returns 422
  (Pydantic) and never reaches the model.
* ``text`` length constraints: empty and over-long text return 422.
* Invalid ``customer_plan`` returns 422.
* 503 when the category model is not loaded: the response body
  documents the cause (``detail`` field).
* Batch happy path: response contains ``count`` and ``latency_ms``
  and one :class:`ClassifyResponse` per item; all rows are persisted.
* Batch size cap: more than ``MAX_BATCH_SIZE`` items returns 422.
* Empty batch is rejected with 422.
* Persistence-failure resilience: if the DB session raises, the single
  endpoint still returns the prediction with ``persisted=False`` and
  HTTP 200 (we never want a working classifier to take the API down
  because SQLite is locked). We exercise this by overriding
  :func:`ticket_router.db.database.get_db` to raise.

We deliberately use the *real* trained model artifact when it is present
in ``artifacts/`` (skip otherwise), because the Day 15 endpoint exists
to serve real predictions - mocking the classifier would just be
re-testing the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from ticket_router.api import main as api_main
from ticket_router.api.routes import classify as classify_routes
from ticket_router.config import settings
from ticket_router.db.database import get_db
from ticket_router.db.models import Base, Prediction, Ticket
from ticket_router.db import get_engine, get_session_factory
from ticket_router.pipeline.inference import reset_default_pipeline
from ticket_router.schemas import MAX_BATCH_SIZE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def artifact_path() -> Path:
    """Skip the whole module if the trained artifact is missing.

    Day 15's contract is "real predictions over HTTP" - there is no
    point in a mock-driven test suite here. If the artifact is gone,
    the integration is broken anyway.
    """
    p = Path(settings.category_model_path)
    if not p.exists():
        pytest.skip(f"Category model artifact not present at {p}")
    return p


@pytest.fixture()
def in_memory_db():
    """Build a fully isolated in-memory SQLite engine and session factory.

    The session factory is explicitly bound to the same engine that
    holds the schema so writes from request handlers are visible to
    the test's verification queries.
    """
    eng = get_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    factory = get_session_factory("sqlite:///:memory:")
    # Re-bind the factory to the engine that actually has the schema.
    # Otherwise ``get_session_factory("sqlite:///:memory:")`` builds a
    # *new* StaticPool engine - a separate in-memory database - and
    # the request session can't see the tables we just created.
    factory.configure(bind=eng)
    return eng, factory


@pytest.fixture()
def client_with_in_memory_db(artifact_path: Path, in_memory_db, monkeypatch: pytest.MonkeyPatch):
    """Build a FastAPI :class:`TestClient` backed by an in-memory DB.

    The real category artifact is loaded by the lifespan; the DB is a
    private in-memory engine accessible through FastAPI's
    ``dependency_overrides`` mechanism. The lifespan's call to
    ``init_db`` is overridden to a no-op so it does not try to create
    a schema on a *different* in-memory engine.

    Yields ``(client, engine, factory)``. The lifespan is driven by
    the :class:`TestClient` context manager, so each test gets a
    fresh app and a clean DB.
    """
    eng, factory = in_memory_db

    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")

    # The lifespan calls ``api_main.init_db(...)``; replace it with a
    # no-op so the in-memory schema we built above stays the source of
    # truth.
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


# ---------------------------------------------------------------------------
# Happy path: single
# ---------------------------------------------------------------------------


def _sample_payload(**overrides) -> dict:
    base = {
        "text": "I was charged twice for my subscription, this is unacceptable!",
        "customer_plan": "pro",
        "customer_id": "cus_test_1",
    }
    base.update(overrides)
    return base


class TestClassifySingle:
    def test_returns_200_and_classify_response(self, client_with_in_memory_db, in_memory_db):
        client, eng, _ = client_with_in_memory_db
        r = client.post("/classify", json=_sample_payload())
        assert r.status_code == 200, r.text
        body = r.json()

        # Spec fields are present (no ``persisted`` - the public contract
        # in docs/PROJECT_PLAN.md §3 does not include it).
        for k in (
            "ticket_id",
            "category",
            "category_confidence",
            "sentiment",
            "sentiment_scores",
            "priority",
            "priority_score",
            "routed_to",
            "urgency_signals",
            "latency_ms",
        ):
            assert k in body, f"missing {k}"

        # Type and value constraints
        assert body["ticket_id"].startswith("tkt_")
        assert body["category"] in {
            "Billing",
            "Authentication",
            "Bug Report",
            "Feature Request",
            "Technical Setup",
        }
        assert body["sentiment"] in {"Positive", "Neutral", "Frustrated", "Angry"}
        assert body["priority"] in {"P1", "P2", "P3"}
        assert body["routed_to"].endswith("-team")
        assert isinstance(body["urgency_signals"], list)
        assert 0.0 <= body["category_confidence"] <= 1.0
        assert 0 <= body["priority_score"] <= 100
        assert isinstance(body["latency_ms"], int) and body["latency_ms"] >= 0

        # Sentiment scores are bounded
        s = body["sentiment_scores"]
        for k in ("neg", "neu", "pos", "compound"):
            assert k in s
            assert -1.0 <= s[k] <= 1.0

        # Persisted to the DB on the happy path
        from sqlalchemy.orm import sessionmaker

        Session_ = sessionmaker(bind=eng, expire_on_commit=False, autoflush=False)
        with Session_() as s:
            ticket = s.get(Ticket, body["ticket_id"])
            assert ticket is not None
            assert ticket.text == _sample_payload()["text"]

    def test_response_round_trips_to_db(self, client_with_in_memory_db, in_memory_db):
        client, eng, _ = client_with_in_memory_db
        r = client.post(
            "/classify",
            json=_sample_payload(text="Refund my double charge please", customer_id="cus_rt"),
        )
        assert r.status_code == 200
        body = r.json()

        # Use a fresh session against the same in-memory engine.
        from sqlalchemy.orm import sessionmaker

        Session_ = sessionmaker(bind=eng, expire_on_commit=False, autoflush=False)
        with Session_() as s:
            ticket = s.get(Ticket, body["ticket_id"])
            assert ticket is not None
            assert ticket.text == "Refund my double charge please"
            assert ticket.customer_id == "cus_rt"
            assert ticket.customer_plan == "pro"

            pred = s.execute(
                select(Prediction).where(Prediction.ticket_id == body["ticket_id"])
            ).scalar_one()
            assert pred.category == body["category"]
            assert pred.sentiment == body["sentiment"]
            assert pred.priority == body["priority"]
            assert pred.priority_score == body["priority_score"]
            assert pred.routed_to == body["routed_to"]
            assert pred.latency_ms == body["latency_ms"]

    def test_defaults_customer_plan_to_free(self, client_with_in_memory_db, in_memory_db):
        client, eng, _ = client_with_in_memory_db
        r = client.post("/classify", json={"text": "cannot log in to my account"})
        assert r.status_code == 200, r.text

        from sqlalchemy.orm import sessionmaker

        Session_ = sessionmaker(bind=eng, expire_on_commit=False, autoflush=False)
        with Session_() as s:
            ticket = s.get(Ticket, r.json()["ticket_id"])
            assert ticket is not None
            assert ticket.customer_plan == "free"
            assert ticket.customer_id is None

    def test_urgency_signals_persist_as_list(self, client_with_in_memory_db, in_memory_db):
        client, eng, _ = client_with_in_memory_db
        text = "I was charged twice and I want a refund NOW, this is urgent!"
        r = client.post(
            "/classify",
            json={"text": text, "customer_plan": "pro"},
        )
        assert r.status_code == 200
        body = r.json()
        # We don't assert specific signals (the lexicon is locked but
        # brittle to depend on here) - just that any non-empty list
        # round-trips.
        from sqlalchemy.orm import sessionmaker

        Session_ = sessionmaker(bind=eng, expire_on_commit=False, autoflush=False)
        with Session_() as s:
            pred = s.execute(
                select(Prediction).where(Prediction.ticket_id == body["ticket_id"])
            ).scalar_one()
            round_tripped = [seg for seg in (pred.urgency_signals or "").split("\n") if seg]
            assert round_tripped == body["urgency_signals"]

    def test_each_call_gets_unique_ticket_id(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        ids = set()
        for _ in range(5):
            r = client.post("/classify", json=_sample_payload())
            assert r.status_code == 200
            ids.add(r.json()["ticket_id"])
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# Validation errors -> 422
# ---------------------------------------------------------------------------


class TestClassifyValidation:
    def test_empty_text_returns_422(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.post("/classify", json={"text": ""})
        assert r.status_code == 422

    def test_missing_text_returns_422(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.post("/classify", json={"customer_plan": "pro"})
        assert r.status_code == 422

    def test_text_too_long_returns_422(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.post("/classify", json={"text": "a" * 8001})
        assert r.status_code == 422

    def test_invalid_customer_plan_returns_422(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.post(
            "/classify",
            json={"text": "hello", "customer_plan": "platinum"},
        )
        assert r.status_code == 422

    def test_unknown_field_returns_422(self, client_with_in_memory_db):
        """``extra='forbid'`` catches typos like ``"Text"`` vs ``"text"``."""
        client, _, _ = client_with_in_memory_db
        r = client.post(
            "/classify",
            json={"text": "hello", "customre_plan": "pro"},  # typo
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# 503 when the model isn't loaded
# ---------------------------------------------------------------------------


class TestClassifyUnavailable:
    def test_503_when_category_classifier_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """If the lifespan failed to load the model, /classify returns 503."""
        monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path}/unused.db")
        monkeypatch.setattr(
            settings, "category_model_path", str(tmp_path / "missing.joblib")
        )
        monkeypatch.setattr(api_main, "init_db", lambda url=None: None)

        reset_default_pipeline()
        app = api_main.create_app()

        with TestClient(app) as client:
            r = client.post("/classify", json=_sample_payload())

        assert r.status_code == 503
        body = r.json()
        # FastAPI wraps HTTPException detail under "detail"
        assert "Category model" in body["detail"] or "pipeline" in body["detail"].lower()

    def test_health_degraded_when_artifact_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Sanity check: the lifespan path that drives 503 also reports /health as degraded."""
        monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path}/unused.db")
        monkeypatch.setattr(
            settings, "category_model_path", str(tmp_path / "missing.joblib")
        )
        monkeypatch.setattr(api_main, "init_db", lambda url=None: None)

        reset_default_pipeline()
        app = api_main.create_app()
        with TestClient(app) as client:
            r = client.get("/health")
        assert r.status_code == 503
        assert r.json()["model_loaded"] is False


# ---------------------------------------------------------------------------
# Persistence failure resilience
# ---------------------------------------------------------------------------


def test_classify_returns_prediction_when_persistence_fails(
    artifact_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A DB write failure must not take down the API.

    We point :func:`get_db` at a fresh in-memory engine that has *no*
    schema, so the INSERT inside ``save_classification`` raises. The
    endpoint should still return a valid prediction body with
    ``persisted=False`` and HTTP 200 - the caller can decide whether
    to retry, but they got their answer.
    """
    # Empty in-memory engine: no tables -> insert fails.
    empty_engine = get_engine("sqlite:///:memory:")
    empty_factory = get_session_factory("sqlite:///:memory:")
    monkeypatch.setattr(settings, "database_url", "sqlite:///:memory:")
    monkeypatch.setattr(api_main, "init_db", lambda url=None: None)

    def _empty_db():
        s = empty_factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    app = api_main.create_app()
    app.dependency_overrides[get_db] = _empty_db

    reset_default_pipeline()
    with TestClient(app) as client:
        r = client.post("/classify", json=_sample_payload())

    empty_engine.dispose()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["persisted"] is False
    # Real classification still happened
    assert body["category"] in {
        "Billing",
        "Authentication",
        "Bug Report",
        "Feature Request",
        "Technical Setup",
    }
    assert body["ticket_id"].startswith("tkt_")


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestClassifyBatch:
    def test_happy_path_persists_all(self, client_with_in_memory_db, in_memory_db):
        client, eng, _ = client_with_in_memory_db
        payload = {
            "items": [
                {"text": "I was charged twice and want a refund", "customer_plan": "pro"},
                {"text": "cannot log in to my account", "customer_plan": "free"},
                {"text": "app crashes when I open settings", "customer_plan": "enterprise"},
            ]
        }
        r = client.post("/classify/batch", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["count"] == 3
        assert isinstance(body["latency_ms"], int) and body["latency_ms"] >= 0
        assert len(body["results"]) == 3
        for item in body["results"]:
            assert item["category"] in {
                "Billing",
                "Authentication",
                "Bug Report",
                "Feature Request",
                "Technical Setup",
            }
            assert item["ticket_id"].startswith("tkt_")

        # All three rows are in the DB (the public response does not
        # carry ``persisted``; verify via the engine directly).
        from sqlalchemy.orm import sessionmaker

        Session_ = sessionmaker(bind=eng, expire_on_commit=False, autoflush=False)
        with Session_() as s:
            n = s.execute(select(func.count()).select_from(Ticket)).scalar_one()
            assert n == 3

    def test_empty_batch_returns_422(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.post("/classify/batch", json={"items": []})
        assert r.status_code == 422

    def test_oversized_batch_returns_422(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        items = [{"text": f"hello {i}"} for i in range(MAX_BATCH_SIZE + 1)]
        r = client.post("/classify/batch", json={"items": items})
        assert r.status_code == 422

    def test_exactly_max_batch_size_accepted(self, client_with_in_memory_db, in_memory_db):
        client, eng, _ = client_with_in_memory_db
        items = [{"text": f"ticket {i}"} for i in range(MAX_BATCH_SIZE)]
        r = client.post("/classify/batch", json={"items": items})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["count"] == MAX_BATCH_SIZE

    def test_unknown_top_level_field_returns_422(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.post(
            "/classify/batch",
            json={"items": [{"text": "hi"}], "tickets": [{"text": "hi"}]},
        )
        assert r.status_code == 422

    def test_invalid_item_in_batch_returns_422(self, client_with_in_memory_db):
        client, _, _ = client_with_in_memory_db
        r = client.post(
            "/classify/batch",
            json={"items": [{"text": "ok"}, {"text": ""}]},  # second is invalid
        )
        assert r.status_code == 422

    def test_batch_503_when_model_not_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path}/unused.db")
        monkeypatch.setattr(
            settings, "category_model_path", str(tmp_path / "missing.joblib")
        )
        monkeypatch.setattr(api_main, "init_db", lambda url=None: None)
        reset_default_pipeline()
        app = api_main.create_app()
        with TestClient(app) as client:
            r = client.post(
                "/classify/batch", json={"items": [{"text": "hi"}]}
            )
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def _collect_paths(app) -> list[str]:
    """Walk an app's router graph, including any :class:`APIRouter` mounts."""
    from fastapi.routing import APIRoute, APIRouter

    out: list[str] = []
    for r in app.routes:
        if isinstance(r, APIRoute):
            out.append(r.path)
        elif isinstance(r, APIRouter):
            for sub in r.routes:
                if isinstance(sub, APIRoute):
                    out.append(sub.path)
        else:
            # FastAPI >=0.110 wraps included routers in a private
            # ``_IncludedRouter`` dataclass with an ``original_router``
            # attribute. Unwrap and recurse.
            inner = getattr(r, "original_router", None) or getattr(r, "router", None)
            if isinstance(inner, APIRouter):
                for sub in inner.routes:
                    if isinstance(sub, APIRoute):
                        out.append(sub.path)
    return out


def _collect_classify_routes(app) -> list:
    """Return every :class:`APIRoute` under ``/classify*``, including nested mounts."""
    from fastapi.routing import APIRoute, APIRouter

    out: list[APIRoute] = []
    for r in app.routes:
        if isinstance(r, APIRoute) and r.path.startswith("/classify"):
            out.append(r)
        elif isinstance(r, APIRouter):
            for sub in r.routes:
                if isinstance(sub, APIRoute) and sub.path.startswith("/classify"):
                    out.append(sub)
        else:
            inner = getattr(r, "original_router", None) or getattr(r, "router", None)
            if isinstance(inner, APIRouter):
                for sub in inner.routes:
                    if isinstance(sub, APIRoute) and sub.path.startswith("/classify"):
                        out.append(sub)
    return out


def test_classify_routes_are_mounted_on_app() -> None:
    """The Day 15 contract is that ``/classify`` appears in the OpenAPI schema."""
    app = api_main.create_app()
    paths = _collect_paths(app)
    assert "/classify" in paths
    assert "/classify/batch" in paths
    classify_routes_list = _collect_classify_routes(app)
    assert classify_routes_list, "no /classify* routes were registered"
    assert all(getattr(r, "tags", []) == ["classify"] for r in classify_routes_list)


def test_classify_module_exports_router() -> None:
    """``classify_routes.router`` is the FastAPI mount point."""
    from fastapi import APIRouter

    from ticket_router.api.routes.classify import get_pipeline

    assert isinstance(classify_routes.router, APIRouter)
    assert callable(get_pipeline)
