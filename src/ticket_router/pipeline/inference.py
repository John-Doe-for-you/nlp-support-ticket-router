"""End-to-end inference pipeline.

Public API
----------
* ``PredictionResult``  : structured output of :meth:`InferencePipeline.predict`.
* ``InferencePipeline`` : orchestrator wiring the Day 7 category classifier,
                          Day 9 sentiment analyzer, Day 10 priority engine,
                          and Day 11 router into a single ``predict()`` call.
* ``get_default_pipeline``: process-wide singleton accessor (loads the
                          trained model from ``artifacts/`` on first use).
* ``predict``           : module-level convenience wrapper around the
                          default pipeline.

Design notes
------------
The pipeline is intentionally side-effect-free: it returns a
``PredictionResult`` instead of touching the database. Persistence is
the Day 13/15 DB + API layer's job; mixing it in here would make unit
testing painful and would couple the scoring code to SQLAlchemy.

Latency is measured around the whole :meth:`predict` body (not just
the model) because that is what the API SLO is keyed on. ``latency_ms``
is an int (rounded) to match the ``ClassifyResponse`` schema.

Ticket ids are short, URL-safe, monotonic-ish: ``tkt_`` + 8 hex chars
from ``secrets.token_hex``. We deliberately avoid random UUIDs with
dashes so the values look clean in URLs and logs.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from ticket_router.models.category_classifier import (
    CATEGORIES,
    CategoryClassifier,
    CategoryPrediction,
)
from ticket_router.models.priority import (
    PriorityEngine,
    PriorityResult,
    get_default_engine,
)
from ticket_router.models.sentiment import (
    SentimentAnalyzer,
    SentimentResult,
    get_default_analyzer,
)
from ticket_router.routing.router import (
    DEFAULT_TEAM,
    Router,
    route_ticket,
)

if TYPE_CHECKING:
    from ticket_router.schemas import ClassifyRequest, ClassifyResponse


_TICKET_ID_PREFIX: str = "tkt_"
_TICKET_ID_HEX_LEN: int = 4  # 8 hex chars total


def _new_ticket_id() -> str:
    """Return a short, URL-safe ticket id like ``tkt_ab12cd34``."""
    return f"{_TICKET_ID_PREFIX}{secrets.token_hex(_TICKET_ID_HEX_LEN)}"


# Public so the Day 15 API/tests can introspect the id format.
TICKET_ID_PREFIX: str = _TICKET_ID_PREFIX


@dataclass(frozen=True)
class PredictionResult:
    """The full output of a single ``InferencePipeline.predict`` call.

    Mirrors the ``ClassifyResponse`` Pydantic schema but stays a plain
    dataclass so the pipeline can be used without FastAPI in scope
    (e.g. from notebooks, scripts, or the Day 19 eval harness).
    """

    ticket_id: str
    text: str
    customer_plan: str
    customer_id: str | None
    category: str
    category_confidence: float
    sentiment: str
    sentiment_scores: dict[str, float]
    priority: str
    priority_score: int
    priority_breakdown: dict[str, float]
    routed_to: str
    urgency_signals: list[str]
    latency_ms: int
    extras: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a dict shaped to match ``ClassifyResponse.model_dump``."""
        return {
            "ticket_id": self.ticket_id,
            "category": self.category,
            "category_confidence": float(self.category_confidence),
            "sentiment": self.sentiment,
            "sentiment_scores": dict(self.sentiment_scores),
            "priority": self.priority,
            "priority_score": int(self.priority_score),
            "routed_to": self.routed_to,
            "urgency_signals": list(self.urgency_signals),
            "latency_ms": int(self.latency_ms),
        }

    def to_response(self) -> "ClassifyResponse":
        """Build a ``ClassifyResponse`` Pydantic model from this result.

        ``customer_id`` and the priority breakdown are intentionally not
        part of the public response schema, so they are dropped here.
        """
        from ticket_router.schemas import ClassifyResponse, SentimentScores

        return ClassifyResponse(
            ticket_id=self.ticket_id,
            category=self.category,  # type: ignore[arg-type]
            category_confidence=float(self.category_confidence),
            sentiment=self.sentiment,  # type: ignore[arg-type]
            sentiment_scores=SentimentScores(**self.sentiment_scores),
            priority=self.priority,  # type: ignore[arg-type]
            priority_score=int(self.priority_score),
            routed_to=self.routed_to,
            urgency_signals=list(self.urgency_signals),
            latency_ms=int(self.latency_ms),
        )


class InferencePipeline:
    """Stateless orchestrator that scores a ticket end-to-end.

    The pipeline is safe to share across threads *as long as* the
    injected ``CategoryClassifier`` is too (which it is - sklearn
    Pipelines built from ``liblinear`` LogReg and TF-IDF vectorizers
    are read-only at predict time).

    Components default to the process-wide singletons from each model
    module, so a bare ``InferencePipeline()`` works in production. Tests
    can inject mocks or freshly-built tiny classifiers via the
    constructor for fast, deterministic runs.
    """

    def __init__(
        self,
        *,
        category_classifier: CategoryClassifier | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        priority_engine: PriorityEngine | None = None,
        router: Router | None = None,
        ticket_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if ticket_id_factory is None:
            ticket_id_factory = _new_ticket_id
        self._ticket_id_factory = ticket_id_factory
        self.category_classifier = category_classifier
        self.sentiment_analyzer = sentiment_analyzer
        self.priority_engine = priority_engine
        self.router = router if router is not None else Router()

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def load_category_model(self, path: str | Path) -> None:
        """Load the trained category model from ``path`` and cache it.

        Called automatically by :meth:`predict` if no classifier was
        injected at construction time. Exposed publicly so the Day 14
        FastAPI lifespan can warm the pipeline at startup.
        """
        self.category_classifier = CategoryClassifier.load(path)

    def _ensure_classifier(self) -> CategoryClassifier:
        clf = self.category_classifier
        if clf is None:
            raise RuntimeError(
                "CategoryClassifier is not loaded. Either inject one at "
                "InferencePipeline(...) or call load_category_model(path)."
            )
        return clf

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        text: str,
        *,
        customer_plan: str = "free",
        customer_id: str | None = None,
        ticket_id: str | None = None,
    ) -> PredictionResult:
        """Score a single ticket end-to-end.

        Parameters
        ----------
        text
            Raw support-ticket text. The classifier's own pipeline
            cleans it; we don't pre-clean here so the API's stored
            ``text`` is the original user input.
        customer_plan
            One of ``"free"``, ``"pro"``, ``"enterprise"`` (validated
            by the Day 10 priority engine).
        customer_id
            Optional opaque customer id (echoed in the result for
            downstream persistence; not part of the public response).
        ticket_id
            Optional pre-generated id. If omitted, a fresh one is
            minted from the configured factory.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        t0 = time.perf_counter()

        classifier = self._ensure_classifier()
        sentiment_analyzer = self.sentiment_analyzer or get_default_analyzer()
        priority_engine = self.priority_engine or get_default_engine()

        # Category
        category_pred: CategoryPrediction = classifier.predict_with_confidence(text)[0]

        # Sentiment (use raw text so VADER picks up exclamation marks etc.)
        sentiment_res: SentimentResult = sentiment_analyzer.analyze(text)

        # Priority
        priority_res: PriorityResult = priority_engine.score(
            sentiment=sentiment_res,
            category_confidence=category_pred.confidence,
            customer_plan=customer_plan,
        )

        # Routing
        team = self.router.route(category_pred.category)

        latency_ms = int(round((time.perf_counter() - t0) * 1000.0))
        tid = ticket_id if ticket_id is not None else self._ticket_id_factory()

        return PredictionResult(
            ticket_id=tid,
            text=text,
            customer_plan=customer_plan,
            customer_id=customer_id,
            category=category_pred.category,
            category_confidence=float(category_pred.confidence),
            sentiment=sentiment_res.label,
            sentiment_scores=sentiment_res.scores.to_dict(),
            priority=priority_res.level,
            priority_score=int(priority_res.score),
            priority_breakdown=priority_res.breakdown.to_dict(),
            routed_to=team,
            urgency_signals=list(sentiment_res.urgency_signals),
            latency_ms=latency_ms,
        )

    def predict_batch(
        self,
        items: Sequence[tuple[str, str] | tuple[str, str, str | None]],
    ) -> list[PredictionResult]:
        """Score multiple tickets sequentially.

        Each item is ``(text, customer_plan)`` or
        ``(text, customer_plan, customer_id)``. Latency is reported per
        ticket (not amortized) so the caller can spot outliers.
        """
        out: list[PredictionResult] = []
        for item in items:
            text = item[0]
            plan = item[1]
            cid = item[2] if len(item) > 2 else None
            out.append(
                self.predict(text, customer_plan=plan, customer_id=cid)
            )
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def classify_request(self, request: "ClassifyRequest") -> PredictionResult:
        """Score a :class:`ClassifyRequest` Pydantic model.

        Mirrors the Day 15 API's request shape so handlers can be a
        one-liner: ``return pipeline.classify_request(req).to_response()``.
        """
        return self.predict(
            text=request.text,
            customer_plan=request.customer_plan,
            customer_id=request.customer_id,
        )


# ----------------------------------------------------------------------
# Module-level singleton
# ----------------------------------------------------------------------

_DEFAULT: InferencePipeline | None = None


def get_default_pipeline() -> InferencePipeline:
    """Return a process-wide :class:`InferencePipeline`.

    The category classifier is loaded lazily on the first :meth:`predict`
    call (or earlier by the Day 14 FastAPI lifespan). All other
    components come from their own module-level singletons.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = InferencePipeline()
    return _DEFAULT


def reset_default_pipeline() -> None:
    """Drop the cached default pipeline. Test helper."""
    global _DEFAULT
    _DEFAULT = None


def predict(
    text: str,
    *,
    customer_plan: str = "free",
    customer_id: str | None = None,
) -> PredictionResult:
    """Convenience wrapper using the default pipeline."""
    return get_default_pipeline().predict(
        text,
        customer_plan=customer_plan,
        customer_id=customer_id,
    )


__all__ = [
    "PredictionResult",
    "InferencePipeline",
    "get_default_pipeline",
    "reset_default_pipeline",
    "predict",
    "TICKET_ID_PREFIX",
]


# Re-export a couple of routing constants so callers don't have to
# import the routing module separately.
__all__ += ["CATEGORIES", "DEFAULT_TEAM", "route_ticket"]
