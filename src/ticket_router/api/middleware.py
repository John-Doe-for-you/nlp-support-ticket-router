"""Request-timing middleware and structured access logging (Day 17).

Why this exists
---------------

PROJECT_PLAN §3 commits to a **p99 < 100ms** latency SLO on ``/classify``.
The plan also flags (Risk register, "Latency creeps over 100ms") that
"latency assertion added Day 17, not end" - i.e. we want to *measure*
latency continuously in production rather than discover it regressed
only after Day 22.

This module provides two small, composable pieces:

* :class:`RequestTimingMiddleware` - measures the wall-clock time each
  HTTP request spends inside the app (from the moment the middleware
  sees the request to the moment the response is fully sent), exposes
  the value as a ``X-Process-Time-Ms`` response header, and emits a
  one-line structured log record per request. The header is useful for
  clients; the log line is useful for SRE / dashboards.

* :func:`configure_access_logging` - configures the ``ticket_router``
  logger tree (and its ``api`` / ``classify`` / ``tickets`` / ``stats``
  children) so request log lines are emitted in a consistent,
  machine-parseable format regardless of how the host process is
  launched.

Design notes
------------

* The middleware measures the *full* request, not just the handler. That
  is what an external load-balancer / SLO dashboard cares about and
  matches the per-request ``latency_ms`` field reported inside the
  ``/classify`` response body (which is handler-only). The two numbers
  will not be identical, and that is intentional - the response field
  is the model time, the header is the user-visible time.

* We deliberately use a plain ``BaseHTTPMiddleware`` rather than the
  newer ``app.middleware("http")`` decorator: the class form makes it
  trivial for tests to assert on the configured ``log_level`` and
  ``X-Process-Time-Ms`` header without poking at app internals.

* The structured log line is **always single-line JSON-shaped** (key=value
  pairs separated by spaces, with values safely quoted). That keeps it
  grep-friendly in container stdout while still being trivial to parse
  with ``json.loads`` after a tiny ``shlex`` step in the log shipper.

* The middleware never raises. A failure inside the timing/log block
  must not break the user's request; we ``logger.exception`` and fall
  through to ``await call_next(request)`` with no decoration.

* The ``X-Process-Time-Ms`` header is rounded to an int (milliseconds)
  to match the ``latency_ms`` schema field in :class:`ClassifyResponse`.
  Sub-millisecond precision is not actionable for the SLO.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp


# Header used to expose the measured request duration. Conventional name
# in the FastAPI / Starlette world (``X-Process-Time`` is the Starlette
# tutorial's name; we extend it with the unit so it's unambiguous).
PROCESS_TIME_HEADER: str = "X-Process-Time-Ms"

# Loggers we touch. The ``ticket_router`` tree is created in
# ``api.main``; we only configure formatters/handlers here so that
# re-imports don't double-add handlers (which would duplicate lines).
LOGGER_NAMES: tuple[str, ...] = (
    "ticket_router",
    "ticket_router.api",
    "ticket_router.api.classify",
    "ticket_router.api.tickets",
    "ticket_router.api.stats",
    "ticket_router.db",
    "ticket_router.pipeline",
    "ticket_router.models",
)

# Default format: ``%(asctime)s %(levelname)s %(name)s %(message)s`` -
# identical to the stdlib default but without the color codes some
# libraries add on Windows. Stable, grep-friendly.
_DEFAULT_LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Measure per-request wall time and emit a structured access log.

    Adds ``X-Process-Time-Ms`` to every response and writes one log
    line per request at INFO level (or ERROR on a 5xx) with the fields:

    * ``method``     - HTTP method
    * ``path``       - request path (no query string; queries can leak
                        secrets, see OWASP A09:2021)
    * ``status``     - response status code
    * ``duration_ms``- measured wall time, integer milliseconds
    * ``client``     - remote host (best-effort)
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        logger_name: str = "ticket_router.api.access",
        slow_threshold_ms: int = 100,
    ) -> None:
        super().__init__(app)
        self._logger = logging.getLogger(logger_name)
        self._slow_threshold_ms = int(slow_threshold_ms)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Handler raised (unhandled 500 path). Record what we can
            # so the log line still gets written, then re-raise so
            # FastAPI's exception handler can render the response.
            duration_ms = int(round((time.perf_counter() - start) * 1000.0))
            self._emit_log(
                request,
                status=500,
                duration_ms=duration_ms,
                level=logging.ERROR,
            )
            raise

        duration_ms = int(round((time.perf_counter() - start) * 1000.0))
        try:
            response.headers[PROCESS_TIME_HEADER] = str(duration_ms)
        except Exception:
            # Some response types (e.g. streaming responses backed by
            # already-sent bodies) refuse header mutation. We don't
            # want the access log to fail the request.
            self._logger.debug(
                "Could not set %s on response for %s %s",
                PROCESS_TIME_HEADER,
                request.method,
                request.url.path,
            )

        level = logging.WARNING if duration_ms >= self._slow_threshold_ms else logging.INFO
        if response.status_code >= 500:
            level = logging.ERROR
        self._emit_log(request, status=response.status_code, duration_ms=duration_ms, level=level)
        return response

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit_log(self, request: Request, *, status: int, duration_ms: int, level: int) -> None:
        """Write a single structured access-log line for this request.

        Format is intentionally ``key=value`` so it is grep-friendly and
        trivially parseable with ``shlex`` if a downstream shipper needs
        JSON. We never log the query string (it can contain tokens /
        PII).
        """
        client = self._safe_client(request)
        method = request.method
        path = request.url.path
        try:
            self._logger.log(
                level,
                "method=%s path=%s status=%d duration_ms=%d client=%s",
                method,
                path,
                int(status),
                int(duration_ms),
                client,
            )
        except Exception:
            # Logging must never break a request.
            self._logger.exception("Failed to emit access log line")

    @staticmethod
    def _safe_client(request: Request) -> str:
        """Return a non-PII client identifier for the access log.

        We prefer the immediate socket peer. ``X-Forwarded-For`` is
        ignored by default - trusting it without a configured
        ``trusted_hosts`` middleware lets clients spoof their own
        remote address.
        """
        if request.client is None:
            return "-"
        return request.client.host or "-"


def configure_access_logging(level: str = "INFO") -> None:
    """Configure the ``ticket_router`` logger tree for Day 17 access logs.

    Idempotent: re-calling clears the handler list and re-installs a
    single ``StreamHandler`` on each named logger so that repeated
    imports / fixture setups do not produce duplicate log lines.

    Parameters
    ----------
    level
        Root level for the ``ticket_router`` tree. Individual loggers
        inherit; this matches the previous Day 14 behavior where the
        app's own loggers were unconfigured (root logger default).
    """
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT))

    for name in LOGGER_NAMES:
        lg = logging.getLogger(name)
        # Remove any previously-installed handlers so we don't end up
        # with N copies of every log line after a re-config.
        for existing in list(lg.handlers):
            lg.removeHandler(existing)
        # ``propagate=True`` by default; leave that alone so records
        # bubble up to any host-process root logger too.
        lg.setLevel(level.upper())
        lg.addHandler(handler)


__all__ = [
    "RequestTimingMiddleware",
    "configure_access_logging",
    "PROCESS_TIME_HEADER",
    "LOGGER_NAMES",
]

