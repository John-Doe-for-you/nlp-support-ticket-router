"""FastAPI application factory and lifespan.

Public API
----------
* ``create_app``  : build a fully-configured :class:`fastapi.FastAPI`
                     instance. Tests call this directly; production
                     (``uvicorn ticket_router.api.main:app``) imports the
                     module-level :data:`app` singleton.
* ``app``         : the production singleton, ready for ``uvicorn``.

Design notes
------------
The app factory pattern is used for two reasons:

1. **Tests** want to spin up an isolated app per test (in-memory DB,
   custom model path) without touching the production singleton. They
   get that by calling :func:`create_app` and using FastAPI's
   ``dependency_overrides`` together with ``TestClient``.
2. **Day 17's** latency middleware and Day 16's stats router can both
   read pipeline/DB state from ``app.state`` instead of importing
   module-level globals, which is the FastAPI-blessed way to share
   state across requests.

Lifespan responsibilities (PROJECT_PLAN Day 14):

* Load the trained category classifier from the configured path and
  stash it on the pipeline + ``app.state.pipeline``.
* Initialize the SQLAlchemy schema (idempotent).
* Log a one-line startup summary so container logs make the boot
  state obvious at a glance.

A failure to load the model does **not** crash the app: ``/health``
returns 503 with ``model_loaded=False`` and the Day 15 ``/classify``
endpoint will surface a clean 503. This keeps the API process
restartable via the orchestrator (Docker / k8s) rather than
flapping in a crash loop when artifacts haven't been baked into
the image yet.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ticket_router.config import settings
from ticket_router.db.database import init_db
from ticket_router.pipeline.inference import InferencePipeline

logger = logging.getLogger("ticket_router.api")

API_TITLE = "NLP Support Ticket Router"
API_DESCRIPTION = (
    "Classifies incoming customer support tickets, scores sentiment and "
    "priority, and routes them to the right team. Built with FastAPI + "
    "SQLAlchemy on top of a TF-IDF + LogReg category classifier, VADER "
    "sentiment, and a rule-based priority engine."
)
API_VERSION = "0.1.0"


def _load_category_model(pipeline: InferencePipeline, path: str) -> bool:
    """Try to load the category model from ``path`` into ``pipeline``.

    Returns ``True`` on success, ``False`` if the artifact is missing or
    unloadable. Never raises: the API must boot so ``/health`` can
    report the failure to a load balancer / orchestrator.
    """
    model_path = Path(path)
    if not model_path.exists():
        logger.warning(
            "Category model artifact not found at %s; /classify will return 503 "
            "until the file is provided (run scripts/train_category.py).",
            model_path,
        )
        return False
    try:
        pipeline.load_category_model(model_path)
    except Exception:
        logger.exception("Failed to load category model from %s", model_path)
        # Drop the half-loaded classifier so the pipeline stays consistent.
        pipeline.category_classifier = None
        return False
    logger.info("Loaded category model from %s", model_path)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: warm pipeline + DB on startup, log on shutdown."""
    logger.info(
        "Starting %s v%s (env=%s, log_level=%s)",
        API_TITLE,
        API_VERSION,
        settings.app_env,
        settings.log_level,
    )
    logger.info("Database URL: %s", settings.database_url)
    logger.info("Category model path: %s", settings.category_model_path)

    # --- Pipeline + model loading ---
    pipeline = InferencePipeline()
    app.state.pipeline = pipeline
    app.state.model_loaded = _load_category_model(pipeline, settings.category_model_path)

    # --- DB schema bootstrap ---
    # Pass the configured URL explicitly so we don't share the cached
    # engine from a previous test/process with a stale URL.
    try:
        init_db(url=settings.database_url)
        app.state.db_ready = True
        logger.info("Database schema ready")
    except Exception:
        logger.exception("Failed to initialize database schema")
        app.state.db_ready = False

    try:
        yield
    finally:
        logger.info("Shutting down %s", API_TITLE)


def create_app() -> FastAPI:
    """Build a :class:`FastAPI` instance with lifespan, CORS, and routes wired.

    CORS is wide-open by default: this is a local-dev / portfolio
    project. Tighten ``allow_origins`` before exposing publicly.
    """
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        # OpenAPI tags appear in /docs; left as the FastAPI default.
    )

    # CORS: open by default for local development. The /docs and
    # /openapi.json endpoints are needed by tools like Swagger UI
    # running on a different port.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, object]:
        """Root: API banner + pointer to OpenAPI docs."""
        return {
            "name": API_TITLE,
            "version": API_VERSION,
            "description": API_DESCRIPTION,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
        }

    @app.get("/health", tags=["meta"], summary="Liveness + readiness probe")
    def health() -> dict[str, object]:
        """Report process liveness, model readiness, and DB readiness.

        Returns HTTP 200 when the service is fully ready to serve
        ``/classify``, HTTP 503 when any required component is down.
        """
        model_loaded = bool(getattr(app.state, "model_loaded", False))
        db_ready = bool(getattr(app.state, "db_ready", False))
        ready = model_loaded and db_ready
        payload: dict[str, object] = {
            "status": "ok" if ready else "degraded",
            "version": API_VERSION,
            "model_loaded": model_loaded,
            "db_ready": db_ready,
            "model_path": settings.category_model_path,
            "database_url": _safe_db_url(settings.database_url),
        }
        if not ready:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=503, content=payload)
        return payload

    # Route modules. Day 15 wires up /classify; Day 16 will add
    # /tickets and /stats. Each router is mounted with its own
    # ``prefix`` (defined inside the module) so the URLs stay
    # namespaced cleanly.
    from ticket_router.api.routes import classify as classify_routes

    app.include_router(classify_routes.router)

    return app


def _safe_db_url(url: str) -> str:
    """Strip the password from a SQLAlchemy URL for safe display in /health."""
    if "@" not in url:
        return url
    scheme_user, _, host_part = url.rpartition("@")
    if "://" not in scheme_user:
        return url
    scheme, _, userinfo = scheme_user.partition("://")
    if ":" in userinfo:
        user, _, _ = userinfo.partition(":")
        userinfo = f"{user}:***"
    return f"{scheme}://{userinfo}@{host_part}"


# Module-level singleton for ``uvicorn ticket_router.api.main:app``.
# Re-created lazily if a test mutates :data:`app` (we don't, by
# convention: tests build their own via :func:`create_app`).
app: FastAPI = create_app()


__all__ = ["create_app", "app", "API_TITLE", "API_VERSION", "lifespan"]
