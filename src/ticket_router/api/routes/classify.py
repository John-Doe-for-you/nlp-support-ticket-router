"""``POST /classify`` and ``POST /classify/batch`` routes (Day 15).

Single-ticket contract
----------------------

::

    POST /classify
    { "text": "...", "customer_plan": "pro", "customer_id": "cus_123" }

returns the :class:`ClassifyResponse` documented in
``docs/PROJECT_PLAN.md`` §3 and persists the result to the ``tickets``
and ``predictions`` tables in the same transaction.

Batch contract
--------------

::

    POST /classify/batch
    { "items": [ {"text": "..."}, {"text": "..."} ] }

returns a :class:`BatchClassifyResponse` with the per-ticket results
and a total ``latency_ms``. Hard-capped at
:data:`ticket_router.schemas.MAX_BATCH_SIZE` to keep the p99 SLO safe.

Design notes
------------

* The pipeline is loaded once by the Day 14 lifespan and shared across
  requests (``app.state.pipeline``). sklearn Pipelines built from
  ``liblinear`` LogReg + TF-IDF are read-only at predict time, so a
  single instance is safe to share across threads.
* Persistence is best-effort: a DB error during save does *not* lose
  the prediction for the caller. We return the prediction with a
  ``persisted=False`` flag in that case and log the exception. The
  orchestrator can choose to retry, but the user-facing API contract
  is still satisfied. (Retries would put a duplicate-id footgun on the
  client; we let Day 19's eval harness surface those failures instead.)
* 503 is returned when the category model is not loaded (the lifespan
  failed). 422 is the Pydantic default for bad payloads.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ticket_router.db.database import get_db
from ticket_router.db.repository import save_classification
from ticket_router.pipeline.inference import (
    InferencePipeline,
    PredictionResult,
)
from ticket_router.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
)

logger = logging.getLogger("ticket_router.api.classify")

router = APIRouter(prefix="/classify", tags=["classify"])


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


def get_pipeline(request: Request) -> InferencePipeline:
    """Return the process-wide :class:`InferencePipeline` stashed on app.state.

    Raises 503 if the lifespan failed to load the category model. We
    deliberately raise here (not return ``None``) so the route handler
    can stay short and FastAPI's exception handler renders a clean
    response.
    """
    pipeline: InferencePipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference pipeline is not initialized",
        )
    if pipeline.category_classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Category model is not loaded. Train or mount the artifact first.",
        )
    return pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _persist_result(session: Session, result: PredictionResult) -> bool:
    """Save ``result`` to the DB.

    Returns ``True`` on success, ``False`` if persistence failed for
    any reason. The caller is expected to log + surface a
    ``persisted`` flag in the response.
    """
    try:
        save_classification(session, result, commit=True)
        return True
    except Exception:
        logger.exception(
            "Failed to persist classification for ticket_id=%s", result.ticket_id
        )
        try:
            session.rollback()
        except Exception:
            logger.exception("Session rollback also failed for ticket_id=%s", result.ticket_id)
        return False


def _to_response_dict(result: PredictionResult, persisted: bool) -> dict[str, Any]:
    """Serialize ``result`` to a JSON-safe dict, with a non-spec ``persisted`` flag.

    ``persisted`` is a Day 15-only debug aid; the published
    :class:`ClassifyResponse` does not include it. We attach it as an
    extra field so the test suite (and curl users) can tell when
    persistence silently failed.
    """
    base = result.to_dict()
    base["persisted"] = persisted
    return base


# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify a single support ticket",
)
def classify_ticket(
    payload: ClassifyRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),
    session: Session = Depends(get_db),
) -> ClassifyResponse | JSONResponse:
    """Score a ticket, persist the result, and return the public response.

    Returns 200 on the happy path. Returns 200-with-``persisted=False``
    on a persistence failure (the caller still gets a useful answer).
    """
    t0 = time.perf_counter()
    result = pipeline.classify_request(payload)
    persisted = _persist_result(session, result)
    total_ms = int(round((time.perf_counter() - t0) * 1000.0))

    if not persisted:
        logger.warning(
            "Returned classification without persistence (ticket_id=%s, "
            "pipeline_latency_ms=%d, total_latency_ms=%d)",
            result.ticket_id,
            result.latency_ms,
            total_ms,
        )
        # The published schema forbids extra fields, so we return a raw
        # JSONResponse for this branch. The body still matches the
        # contract plus the ``persisted`` flag.
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=_to_response_dict(result, persisted=False),
        )

    return result.to_response()


# ---------------------------------------------------------------------------
# POST /classify/batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=BatchClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify up to MAX_BATCH_SIZE tickets in one request",
)
def classify_batch(
    payload: BatchClassifyRequest,
    pipeline: InferencePipeline = Depends(get_pipeline),
    session: Session = Depends(get_db),
) -> BatchClassifyResponse:
    """Score each item sequentially and persist the results.

    Batch latency is reported as the wall-clock for the whole handler
    (including persistence), not the sum of per-ticket latencies - that
    is what a load-balancer or SLO dashboard actually cares about.
    """
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    persisted_count = 0
    for item in payload.items:
        result = pipeline.classify_request(item)
        persisted = _persist_result(session, result)
        if persisted:
            persisted_count += 1
        results.append(_to_response_dict(result, persisted=persisted))

    total_ms = int(round((time.perf_counter() - t0) * 1000.0))
    if persisted_count != len(results):
        logger.warning(
            "Batch completed with partial persistence: %d/%d saved (total_latency_ms=%d)",
            persisted_count,
            len(results),
            total_ms,
        )

    return BatchClassifyResponse(
        count=len(results),
        latency_ms=total_ms,
        results=[ClassifyResponse(**r) for r in results],  # type: ignore[arg-type]
    )


__all__ = ["router", "get_pipeline"]
