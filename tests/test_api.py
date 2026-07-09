"""Tests for the Day 14 FastAPI application shell.

Covers the public surface added today:

* ``create_app`` builds a working :class:`FastAPI` instance.
* ``GET /`` returns the API banner with version + pointers to docs.
* ``GET /health`` reports model + DB readiness and degrades to 503
  when the model artifact is missing.
* The lifespan actually loads the trained category classifier into
  ``app.state.pipeline`` and stamps ``app.state.model_loaded``.
* ``_safe_db_url`` redacts passwords for the ``/health`` response.
* The OpenAPI schema is exposed at ``/openapi.json``.

We deliberately avoid touching the on-disk ``tickets.db`` and the real
``artifacts/category_model.joblib``: each test points ``settings`` at a
temp file before constructing the app, then restores the originals.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ticket_router.api import main as api_main
from ticket_router.config import settings
from ticket_router.pipeline.inference import InferencePipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a FastAPI app pointed at a temp DB and a real-but-copied model.

    The category artifact is large-ish (~MB); instead of copying, we point
    ``settings.category_model_path`` at the on-disk artifact if it exists
    and skip the test otherwise. ``settings.database_url`` is always
    redirected to a tempfile in ``tmp_path`` so we never touch the
    checked-out ``tickets.db``.
    """
    db_path = (tmp_path / "tickets.db").resolve()
    # ``sqlite:///<abs-path>`` (three slashes) is the form SQLAlchemy
    # expects for a file-backed DB at an absolute path. Without the
    # leading triple-slash it interprets the path as relative-to-CWD.
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{db_path}")

    artifact = Path(settings.category_model_path)
    if not artifact.exists():
        pytest.skip(f"Category model artifact not present at {artifact}")

    # Reset module-level cached pipeline (and any other module state) so
    # each test starts from a clean slate even if a prior test built an
    # app that warmed the default pipeline.
    from ticket_router.pipeline.inference import reset_default_pipeline

    reset_default_pipeline()

    app = api_main.create_app()
    return app, db_path


# ---------------------------------------------------------------------------
# Root + health
# ---------------------------------------------------------------------------


def test_root_returns_api_banner(isolated_app) -> None:
    app, _ = isolated_app
    with TestClient(app) as client:
        r = client.get("/")

    assert r.status_code == 200
    body = r.json()
    assert body["name"] == api_main.API_TITLE
    assert body["version"] == api_main.API_VERSION
    assert body["docs"] == "/docs"
    assert body["openapi"] == "/openapi.json"
    assert body["health"] == "/health"


def test_root_not_in_openapi_schema(isolated_app) -> None:
    """The root banner is for humans; keep /docs focused on real endpoints."""
    app, _ = isolated_app
    with TestClient(app) as client:
        r = client.get("/openapi.json")

    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/" not in paths
    assert "/health" in paths


def test_health_ok_when_ready(isolated_app) -> None:
    app, _ = isolated_app
    with TestClient(app) as client:
        r = client.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["db_ready"] is True
    assert body["version"] == api_main.API_VERSION
    assert body["model_path"] == settings.category_model_path
    # The default SQLite URL has no password, so the redactor passes it
    # through unchanged.
    assert body["database_url"] == settings.database_url


def test_health_503_when_model_artifact_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = (tmp_path / "tickets.db").resolve()
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(settings, "category_model_path", str(tmp_path / "nope.joblib"))

    from ticket_router.pipeline.inference import reset_default_pipeline

    reset_default_pipeline()

    app = api_main.create_app()
    with TestClient(app) as client:
        r = client.get("/health")

    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["db_ready"] is True  # DB init still succeeds


def test_health_503_when_model_load_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupted artifact must not crash the app; /health must say so."""
    db_path = (tmp_path / "tickets.db").resolve()
    bad_artifact = tmp_path / "broken.joblib"
    bad_artifact.write_bytes(b"this is not a joblib file")

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(settings, "category_model_path", str(bad_artifact))

    from ticket_router.pipeline.inference import reset_default_pipeline

    reset_default_pipeline()

    app = api_main.create_app()
    with TestClient(app) as client:
        r = client.get("/health")

    assert r.status_code == 503
    body = r.json()
    assert body["model_loaded"] is False
    assert body["db_ready"] is True


# ---------------------------------------------------------------------------
# Lifespan / startup behavior
# ---------------------------------------------------------------------------


def test_lifespan_loads_model_into_app_state(isolated_app) -> None:
    app, _ = isolated_app
    with TestClient(app) as client:
        # Hitting any endpoint forces the lifespan to run (TestClient
        # runs startup before the first request and shutdown on exit).
        client.get("/health")
        pipeline = app.state.pipeline
        loaded_flag = app.state.model_loaded

    assert isinstance(pipeline, InferencePipeline)
    assert loaded_flag is True
    assert pipeline.category_classifier is not None
    # The classifier is the trained artifact and exposes the locked
    # five categories.
    assert tuple(pipeline.category_classifier.classes_()) == (
        "Authentication",
        "Billing",
        "Bug Report",
        "Feature Request",
        "Technical Setup",
    )


def test_lifespan_initializes_db_schema(isolated_app) -> None:
    app, db_path = isolated_app
    with TestClient(app) as client:
        client.get("/health")

    assert db_path.exists()
    # SQLite header check: a real DB file starts with the magic string.
    with open(db_path, "rb") as fh:
        header = fh.read(15)
    assert header.startswith(b"SQLite format 3")


def test_lifespan_swallows_model_load_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing artifact must not crash startup."""
    db_path = (tmp_path / "tickets.db").resolve()
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(settings, "category_model_path", str(tmp_path / "missing.joblib"))

    from ticket_router.pipeline.inference import reset_default_pipeline

    reset_default_pipeline()

    app = api_main.create_app()
    with TestClient(app) as client:
        # Should not raise; should report degraded.
        r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["model_loaded"] is False
    # And the pipeline should still be on app.state, just unloaded.
    assert isinstance(app.state.pipeline, InferencePipeline)
    assert app.state.pipeline.category_classifier is None


# ---------------------------------------------------------------------------
# OpenAPI / docs
# ---------------------------------------------------------------------------


def test_openapi_includes_health_endpoint(isolated_app) -> None:
    app, _ = isolated_app
    with TestClient(app) as client:
        r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"] == api_main.API_TITLE
    assert schema["info"]["version"] == api_main.API_VERSION
    assert "/health" in schema["paths"]
    assert schema["paths"]["/health"]["get"]["summary"]


def test_swagger_ui_loads(isolated_app) -> None:
    app, _ = isolated_app
    with TestClient(app) as client:
        r = client.get("/docs")
    assert r.status_code == 200
    # Swagger UI ships a tiny stub HTML that references the openapi URL.
    assert "swagger" in r.text.lower() or "openapi" in r.text.lower()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected",
    [
        ("sqlite:///./tickets.db", "sqlite:///./tickets.db"),
        ("postgresql://user:secret@db:5432/app", "postgresql://user:***@db:5432/app"),
        ("postgresql://user@db:5432/app", "postgresql://user@db:5432/app"),
        ("mysql+pymysql://alice:hunter2@host/x", "mysql+pymysql://alice:***@host/x"),
    ],
)
def test_safe_db_url_redacts_password(url: str, expected: str) -> None:
    assert api_main._safe_db_url(url) == expected


def test_create_app_returns_fresh_instance_each_call() -> None:
    """create_app() must not share app.state across calls (test isolation)."""
    a = api_main.create_app()
    b = api_main.create_app()
    assert a is not b
    assert isinstance(a, FastAPI)
    assert isinstance(b, FastAPI)


def test_module_level_app_singleton_exists() -> None:
    """``uvicorn ticket_router.api.main:app`` relies on the module singleton."""
    assert isinstance(api_main.app, FastAPI)
    assert api_main.app.title == api_main.API_TITLE


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts with no leaked environment overrides."""
    # pydantic-settings reads .env at import time; clear any TICKET_ROUTER_*
    # vars a sibling process might have set so module-level state is
    # consistent across the suite.
    for key in list(os.environ):
        if key.startswith("TICKET_ROUTER_") or key in {
            "DATABASE_URL",
            "CATEGORY_MODEL_PATH",
        }:
            monkeypatch.delenv(key, raising=False)
