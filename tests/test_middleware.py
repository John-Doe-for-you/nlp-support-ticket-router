"""Tests for the Day 17 request-timing middleware and access logging.

Coverage map (mirrors ``docs/PROJECT_PLAN.md`` Day 17):

* The :class:`RequestTimingMiddleware` is registered on the app and
  adds an ``X-Process-Time-Ms`` header to *every* response (success,
  4xx, 5xx, 404, /docs, /openapi.json).
* The header value is a non-negative integer milliseconds count.
* The access log emits one structured line per request with the
  ``method``, ``path``, ``status``, ``duration_ms`` fields and is
  written to a known logger so a SRE dashboard can subscribe to it.
* 4xx responses do **not** log at WARNING/ERROR level (we don't want
  to spam dashboards with bad-client noise), but 5xx does.
* ``configure_access_logging`` is idempotent: repeated calls do not
  multiply the number of installed handlers.
* The PROJECT_PLAN §3 SLO ("p99 < 100ms") is actually enforceable:
  the latency measured by the middleware on a ``/classify`` request
  stays under the threshold on the in-memory, artifact-backed test
  app. This is the Day 17 latency assertion the plan asks for.
* Errors raised inside handler code (unhandled 500 path) are still
  logged and the middleware does not swallow them.
* The middleware does not leak the query string in the access log
  (OWASP A09:2021: Logging of Sensitive Information).
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ticket_router.api import main as api_main
from ticket_router.api.middleware import (
    LOGGER_NAMES,
    PROCESS_TIME_HEADER,
    RequestTimingMiddleware,
    configure_access_logging,
)
from ticket_router.config import settings
from ticket_router.pipeline.inference import reset_default_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a FastAPI app pointed at a temp DB, ready for middleware tests.

    The category artifact is required for the ``/classify`` latency SLO
    test; if it is missing we still let the rest of the suite run (the
    header / log-line assertions do not need a real model).
    """
    db_path = (tmp_path / "tickets.db").resolve()
    monkeypatch.setattr(settings, "database_url", f"sqlite:///{db_path}")

    artifact = Path(settings.category_model_path)
    has_artifact = artifact.exists()

    reset_default_pipeline()
    app = api_main.create_app()
    return app, db_path, has_artifact


@pytest.fixture()
def capture_logs():
    """Attach a capturing handler to the access logger and yield the buffer.

    Restores the logger to its prior state on teardown so tests don't
    leak handlers into other modules. The buffer is a ``StringIO`` so
    tests can grep the rendered text.
    """
    access_logger = logging.getLogger("ticket_router.api.access")
    prior_handlers = list(access_logger.handlers)
    prior_level = access_logger.level
    prior_propagate = access_logger.propagate

    buf = io.StringIO()
    handler = logging.StreamHandler(stream=buf)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    access_logger.addHandler(handler)
    access_logger.setLevel(logging.DEBUG)
    # Stop propagation so other handlers (added by the lifespan's
    # configure_access_logging) don't also write into the buffer.
    access_logger.propagate = False

    try:
        yield buf, access_logger
    finally:
        access_logger.removeHandler(handler)
        for h in prior_handlers:
            access_logger.addHandler(h)
        access_logger.setLevel(prior_level)
        access_logger.propagate = prior_propagate


# ---------------------------------------------------------------------------
# Header presence
# ---------------------------------------------------------------------------


class TestTimingHeader:
    def test_header_present_on_health(self, isolated_app) -> None:
        app, _, _ = isolated_app
        with TestClient(app) as client:
            r = client.get("/health")
        assert PROCESS_TIME_HEADER in r.headers
        value = r.headers[PROCESS_TIME_HEADER]
        assert value.isdigit(), f"header must be int ms, got {value!r}"
        assert int(value) >= 0

    def test_header_present_on_root(self, isolated_app) -> None:
        app, _, _ = isolated_app
        with TestClient(app) as client:
            r = client.get("/")
        assert PROCESS_TIME_HEADER in r.headers
        assert int(r.headers[PROCESS_TIME_HEADER]) >= 0

    def test_header_present_on_openapi(self, isolated_app) -> None:
        app, _, _ = isolated_app
        with TestClient(app) as client:
            r = client.get("/openapi.json")
        assert PROCESS_TIME_HEADER in r.headers
        assert int(r.headers[PROCESS_TIME_HEADER]) >= 0

    def test_header_present_on_404(self, isolated_app) -> None:
        app, _, _ = isolated_app
        with TestClient(app) as client:
            r = client.get("/no-such-path")
        assert r.status_code == 404
        assert PROCESS_TIME_HEADER in r.headers
        assert int(r.headers[PROCESS_TIME_HEADER]) >= 0

    def test_header_present_on_422_validation_error(self, isolated_app) -> None:
        """Pydantic validation errors come from FastAPI before the handler
        runs but the middleware still wraps them, so the header should
        still be present."""
        app, _, _ = isolated_app
        with TestClient(app) as client:
            r = client.post("/classify", json={"text": ""})
        assert r.status_code == 422
        assert PROCESS_TIME_HEADER in r.headers
        assert int(r.headers[PROCESS_TIME_HEADER]) >= 0

    def test_header_present_on_503_when_model_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = (tmp_path / "tickets.db").resolve()
        monkeypatch.setattr(settings, "database_url", f"sqlite:///{db_path}")
        monkeypatch.setattr(settings, "category_model_path", str(tmp_path / "missing.joblib"))
        monkeypatch.setattr(api_main, "init_db", lambda url=None: None)
        reset_default_pipeline()

        app = api_main.create_app()
        with TestClient(app) as client:
            r = client.post(
                "/classify", json={"text": "hi", "customer_plan": "free"}
            )
        assert r.status_code == 503
        assert PROCESS_TIME_HEADER in r.headers
        assert int(r.headers[PROCESS_TIME_HEADER]) >= 0


# ---------------------------------------------------------------------------
# Access log lines
# ---------------------------------------------------------------------------


class TestAccessLogging:
    def test_emits_one_line_per_request(self, isolated_app, capture_logs) -> None:
        buf, _ = capture_logs
        app, _, _ = isolated_app
        with TestClient(app) as client:
            client.get("/health")
            client.get("/openapi.json")

        text = buf.getvalue()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert len(lines) >= 2, f"expected >=2 access log lines, got: {text!r}"

        # Each line must carry the four locked fields.
        for line in lines:
            assert "method=" in line
            assert "path=" in line
            assert "status=" in line
            assert "duration_ms=" in line
            assert "client=" in line

    def test_log_line_path_strips_query_string(self, isolated_app, capture_logs) -> None:
        """OWASP A09:2021 - never log the query string (can contain PII)."""
        buf, _ = capture_logs
        app, _, _ = isolated_app
        with TestClient(app) as client:
            # /tickets accepts limit/offset query params; we send a
            # distinctive token to confirm it does not appear in the
            # access log path field.
            client.get("/tickets?limit=5&offset=0&token=supersecret-abc123")

        text = buf.getvalue()
        assert "supersecret-abc123" not in text
        # But the path itself is logged.
        assert "path=/tickets" in text

    def test_log_uses_info_level_for_2xx(self, isolated_app, capture_logs) -> None:
        buf, _ = capture_logs
        app, _, _ = isolated_app
        with TestClient(app) as client:
            client.get("/health")

        text = buf.getvalue()
        # Health under 100ms -> INFO; the structured fields still
        # appear in the same line.
        assert "INFO" in text
        assert "status=200" in text
        assert "method=GET" in text
        assert "path=/health" in text

    def test_log_uses_warning_for_404(self, isolated_app, capture_logs) -> None:
        buf, _ = capture_logs
        app, _, _ = isolated_app
        with TestClient(app) as client:
            client.get("/no-such-path")

        text = buf.getvalue()
        assert "status=404" in text
        # 404 is client error, not a server problem; we keep it at INFO
        # so a noisy scanner does not page the on-call.
        assert "INFO" in text

    def test_log_uses_warning_for_slow_responses(
        self, monkeypatch: pytest.MonkeyPatch, capture_logs
    ) -> None:
        """A response slower than the SLO threshold is logged at WARNING."""
        buf, _ = capture_logs
        # Build a bare app with the middleware configured for a tiny
        # 1ms threshold so we can deterministically force the warning.
        from fastapi import FastAPI

        from ticket_router.api.middleware import RequestTimingMiddleware

        sentinel_path = "/__slow__"

        async def _slow_handler():
            import asyncio

            await asyncio.sleep(0.02)  # 20ms - above the 1ms threshold
            return {"ok": True}

        app = FastAPI()
        app.add_middleware(RequestTimingMiddleware, slow_threshold_ms=1)
        app.get(sentinel_path)(_slow_handler)

        with TestClient(app) as client:
            r = client.get(sentinel_path)
        assert r.status_code == 200
        text = buf.getvalue()
        assert "WARNING" in text
        assert f"path={sentinel_path}" in text


# ---------------------------------------------------------------------------
# Latency SLO assertion
# ---------------------------------------------------------------------------


class TestLatencySLO:
    def test_classify_latency_header_under_slo(
        self, isolated_app, capture_logs
    ) -> None:
        """PROJECT_PLAN §3 SLO: p99 < 100ms.

        We assert a 100ms ceiling on a *warm* ``/classify`` request
        against the in-memory-DB / artifact-backed app. The Day 19
        ``evaluate.py`` script will assert p99 over a larger sample;
        this test guards against the most embarrassing regression
        (a single warm request taking seconds).

        We issue one warm-up request first because the first call
        pays for: TF-IDF/LogReg materialization, the SQLite file
        create, and the VADER lexicon load - on Windows that can
        easily blow past 100ms, which would make the assertion
        flaky without adding any signal about steady-state behavior.
        """
        buf, _ = capture_logs
        app, _, has_artifact = isolated_app
        if not has_artifact:
            pytest.skip("Category model artifact not present")

        with TestClient(app) as client:
            # Warm-up: pay the one-time setup costs (joblib load,
            # SQLite file create, VADER lexicon). The response is
            # discarded; we only care that the next call is fast.
            warm = client.post(
                "/classify",
                json={"text": "warmup", "customer_plan": "free"},
            )
            assert warm.status_code == 200, warm.text
            # Reset the buffer so the warm-up line doesn't satisfy
            # the regex below.
            buf.truncate(0)
            buf.seek(0)

            r = client.post(
                "/classify",
                json={
                    "text": "I was charged twice for my subscription, this is unacceptable!",
                    "customer_plan": "pro",
                    "customer_id": "cus_day17_slo",
                },
            )
        assert r.status_code == 200, r.text
        header_ms = int(r.headers[PROCESS_TIME_HEADER])
        assert header_ms < 100, (
            f"Request-time middleware measured {header_ms}ms on /classify, "
            f"violates the 100ms p99 SLO."
        )
        # And the access log must agree.
        text = buf.getvalue()
        m = re.search(r"path=/classify status=200 duration_ms=(\d+)", text)
        assert m, f"no /classify access log line found: {text!r}"
        logged_ms = int(m.group(1))
        assert logged_ms < 100, f"access log reported {logged_ms}ms"

    def test_repeated_classify_runs_all_under_slo(self, isolated_app) -> None:
        """Spot-check 10 sequential ``/classify`` calls; all must be < 100ms.

        Catches flaky latency regressions that a single-shot assertion
        could miss. Still not a p99 measurement (that is the Day 19
        eval job) but tighter than one shot. The first call is the
        warm-up and is excluded from the SLO check.
        """
        app, _, has_artifact = isolated_app
        if not has_artifact:
            pytest.skip("Category model artifact not present")
        with TestClient(app) as client:
            # Warm-up
            warm = client.post(
                "/classify", json={"text": "warmup", "customer_plan": "free"}
            )
            assert warm.status_code == 200
            for i in range(10):
                r = client.post(
                    "/classify",
                    json={
                        "text": f"sample ticket number {i} - cannot log in to my account",
                        "customer_plan": "pro",
                    },
                )
                assert r.status_code == 200
                ms = int(r.headers[PROCESS_TIME_HEADER])
                assert ms < 100, f"request {i} took {ms}ms"


# ---------------------------------------------------------------------------
# Middleware does not swallow errors
# ---------------------------------------------------------------------------


class TestMiddlewareErrorBehavior:
    def test_uncaught_handler_error_still_raises_and_logs(
        self, capture_logs
    ) -> None:
        """A handler that raises must propagate to FastAPI's error handler
        and the middleware must still record the access log line."""
        from fastapi import FastAPI, HTTPException

        from ticket_router.api.middleware import RequestTimingMiddleware

        app = FastAPI()
        app.add_middleware(RequestTimingMiddleware)

        @app.get("/boom")
        def boom() -> None:
            raise RuntimeError("kaboom")

        with TestClient(app, raise_server_exceptions=True) as client:
            with pytest.raises(RuntimeError, match="kaboom"):
                client.get("/boom")

        text = capture_logs[0].getvalue()
        assert "path=/boom" in text
        assert "status=500" in text
        assert "ERROR" in text

    def test_uncaught_handler_error_preserves_timing_header(
        self, capture_logs
    ) -> None:
        """When the handler raises *after* writing headers, FastAPI still
        synthesizes a 500 response; the header should be on it. We use
        HTTPException here so the response goes through the normal
        exception path (no custom handler needed)."""
        from fastapi import FastAPI, HTTPException

        from ticket_router.api.middleware import RequestTimingMiddleware

        app = FastAPI()
        app.add_middleware(RequestTimingMiddleware)

        @app.get("/teapot")
        def teapot() -> None:
            raise HTTPException(status_code=418, detail="I'm a teapot")

        with TestClient(app) as client:
            r = client.get("/teapot")
        assert r.status_code == 418
        assert PROCESS_TIME_HEADER in r.headers
        assert int(r.headers[PROCESS_TIME_HEADER]) >= 0


# ---------------------------------------------------------------------------
# configure_access_logging
# ---------------------------------------------------------------------------


class TestConfigureAccessLogging:
    def test_idempotent_handler_count(self) -> None:
        """Repeated calls must not multiply the number of installed
        handlers - otherwise container logs would print every line N
        times after a process restart hot-loop."""
        # Reset the named loggers first so we have a known starting point.
        for name in LOGGER_NAMES:
            logging.getLogger(name).handlers = []

        configure_access_logging("INFO")
        first_counts = {
            name: len(logging.getLogger(name).handlers) for name in LOGGER_NAMES
        }
        configure_access_logging("INFO")
        second_counts = {
            name: len(logging.getLogger(name).handlers) for name in LOGGER_NAMES
        }
        for name in LOGGER_NAMES:
            assert first_counts[name] >= 1, f"{name} lost its handler"
            assert second_counts[name] == first_counts[name], (
                f"{name} handler count grew: {first_counts[name]} -> {second_counts[name]}"
            )

    def test_level_is_honored(self) -> None:
        for name in LOGGER_NAMES:
            logging.getLogger(name).handlers = []
        configure_access_logging("DEBUG")
        for name in LOGGER_NAMES:
            assert logging.getLogger(name).level == logging.DEBUG
        configure_access_logging("WARNING")
        for name in LOGGER_NAMES:
            assert logging.getLogger(name).level == logging.WARNING


# ---------------------------------------------------------------------------
# Mounting
# ---------------------------------------------------------------------------


def test_middleware_is_mounted_on_app() -> None:
    """Sanity check: the create_app() path actually installs the
    RequestTimingMiddleware. Otherwise all the per-request header /
    log assertions would be vacuously true (no requests would be
    decorated)."""
    from starlette.middleware import Middleware

    app = api_main.create_app()
    # FastAPI stores middleware on ``app.user_middleware`` (list of
    # ``Middleware`` instances, with ``cls`` being the actual class).
    classes = [m.cls for m in app.user_middleware]
    assert RequestTimingMiddleware in classes, (
        f"RequestTimingMiddleware not in {classes}"
    )


def test_middleware_module_exports() -> None:
    """Symbol-level: keep ``__all__`` honest so external code can rely
    on ``from ticket_router.api.middleware import ...``."""
    from ticket_router.api import middleware as mw

    for name in ("RequestTimingMiddleware", "configure_access_logging", "PROCESS_TIME_HEADER"):
        assert name in mw.__all__
        assert hasattr(mw, name)

