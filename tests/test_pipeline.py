"""Tests for the Day 11 end-to-end inference pipeline.

These tests cover:

* the :class:`PredictionResult` dataclass + its ``to_dict`` / ``to_response``
  helpers (no model loading)
* the :class:`InferencePipeline` orchestrator, with a small in-memory
  :class:`CategoryClassifier` so the suite stays fast and deterministic
* the ``get_default_pipeline`` / ``predict`` module-level helpers
* the lazy model-loading behavior

The bulk of the *integration* test coverage (20+ realistic ticket
scenarios, real model artifact, latency assertions) is the explicit
Day 12 deliverable per ``docs/PROJECT_PLAN.md``. This file provides
the structural Day 11 coverage so the orchestrator is never shipped
untested.
"""

from __future__ import annotations

import pytest

from ticket_router.models.category_classifier import (
    CATEGORIES,
    CategoryClassifier,
)
from ticket_router.models.priority import (
    PRIORITY_LEVELS,
    PriorityEngine,
)
from ticket_router.models.sentiment import (
    SENTIMENT_CLASSES,
    SentimentAnalyzer,
)
from ticket_router.pipeline.inference import (
    InferencePipeline,
    PredictionResult,
    TICKET_ID_PREFIX,
    get_default_pipeline,
    predict,
    reset_default_pipeline,
)
from ticket_router.routing.router import ALL_TEAMS, Router, route_ticket


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_SMALL_CORPUS: dict[str, list[str]] = {
    "Billing": [
        "I was charged twice for my subscription this month.",
        "Please refund my last invoice, the payment was duplicated.",
        "My credit card on file was declined for the renewal.",
        "I need an itemized receipt for the plan I paid for.",
        "Why was I overcharged on the subscription fee?",
    ],
    "Authentication": [
        "I can't log in to my account, password reset isn't working.",
        "Two factor authentication code is not arriving on my phone.",
        "My account is locked after too many failed login attempts.",
        "I forgot my password and the reset email never came through.",
        "Single sign on with our SSO provider keeps failing.",
    ],
    "Bug Report": [
        "The app crashes every time I open the settings page.",
        "Getting an internal server error 500 when uploading a file.",
        "After the latest update the dashboard is blank.",
        "The export feature is broken, it freezes mid-download.",
        "Data loss: my saved items disappeared after the sync.",
    ],
    "Feature Request": [
        "Could you add dark mode to the dashboard please.",
        "It would be great if the app supported bulk import from CSV.",
        "Please add an integration with Slack for notifications.",
        "I wish there was a way to schedule reports automatically.",
        "Would love a roadmap view of upcoming features.",
    ],
    "Technical Setup": [
        "How do I install the SDK on a fresh Ubuntu machine?",
        "Step by step guide for configuring the API key in my .env.",
        "Where do I configure the docker container for production?",
        "Tutorial for setting up the kubernetes deployment please.",
        "How can I configure the CI/CD pipeline with GitHub Actions?",
    ],
}


@pytest.fixture(scope="module")
def small_classifier() -> CategoryClassifier:
    clf = CategoryClassifier.build()
    texts: list[str] = []
    labels: list[str] = []
    for cat, docs in _SMALL_CORPUS.items():
        texts.extend(docs)
        labels.extend([cat] * len(docs))
    return clf.fit(texts, labels)


@pytest.fixture()
def pipeline(small_classifier: CategoryClassifier) -> InferencePipeline:
    return InferencePipeline(category_classifier=small_classifier)


# ---------------------------------------------------------------------------
# PredictionResult dataclass
# ---------------------------------------------------------------------------


def _make_result(**overrides: object) -> PredictionResult:
    base: dict[str, object] = {
        "ticket_id": "tkt_abcd1234",
        "text": "sample",
        "customer_plan": "free",
        "customer_id": None,
        "category": "Billing",
        "category_confidence": 0.91,
        "sentiment": "Neutral",
        "sentiment_scores": {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0},
        "priority": "P3",
        "priority_score": 5,
        "priority_breakdown": {
            "urgency": 0.0,
            "sentiment": 0.0,
            "plan": 0.0,
            "confidence": 0.5,
        },
        "routed_to": "billing-team",
        "urgency_signals": [],
        "latency_ms": 12,
    }
    base.update(overrides)
    return PredictionResult(**base)  # type: ignore[arg-type]


def test_prediction_result_to_dict_has_api_response_shape() -> None:
    r = _make_result()
    payload = r.to_dict()
    expected = {
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
    }
    assert set(payload) == expected
    assert payload["ticket_id"] == "tkt_abcd1234"
    assert payload["category"] == "Billing"
    assert payload["sentiment_scores"]["compound"] == 0.0


def test_prediction_result_to_dict_types_are_json_safe() -> None:
    r = _make_result()
    payload = r.to_dict()
    assert isinstance(payload["ticket_id"], str)
    assert isinstance(payload["category_confidence"], float)
    assert isinstance(payload["priority_score"], int)
    assert isinstance(payload["urgency_signals"], list)
    assert isinstance(payload["latency_ms"], int)


def test_prediction_result_to_response_builds_pydantic_model() -> None:
    r = _make_result(category="Bug Report", sentiment="Angry", priority="P1")
    response = r.to_response()
    assert response.ticket_id == "tkt_abcd1234"
    assert response.category == "Bug Report"
    assert response.sentiment == "Angry"
    assert response.priority == "P1"
    assert response.priority_score == 5
    assert response.routed_to == "billing-team"
    # Round-trip via model_dump matches the public schema.
    dump = response.model_dump()
    assert set(dump) == {
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
    }


def test_prediction_result_is_frozen() -> None:
    r = _make_result()
    with pytest.raises(Exception):
        r.category = "Bug Report"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# InferencePipeline - structural
# ---------------------------------------------------------------------------


def test_pipeline_uses_injected_components(small_classifier: CategoryClassifier) -> None:
    p = InferencePipeline(
        category_classifier=small_classifier,
        sentiment_analyzer=SentimentAnalyzer(),
        priority_engine=PriorityEngine(),
        router=Router(),
    )
    assert p.category_classifier is small_classifier
    assert p.sentiment_analyzer is not None
    assert p.priority_engine is not None
    assert p.router is not None


def test_pipeline_default_router_is_fresh_instance() -> None:
    # Two pipelines should not share a Router; if the user customizes one
    # the other must stay on the locked defaults.
    a = InferencePipeline()
    b = InferencePipeline()
    a.router.default_team = "rogue-team"
    assert b.router.default_team == "support-team"


# ---------------------------------------------------------------------------
# InferencePipeline.predict
# ---------------------------------------------------------------------------


def test_predict_returns_a_prediction_result(pipeline: InferencePipeline) -> None:
    r = pipeline.predict("I was charged twice for my subscription!")
    assert isinstance(r, PredictionResult)
    assert r.ticket_id.startswith(TICKET_ID_PREFIX)
    assert len(r.ticket_id) == len(TICKET_ID_PREFIX) + 8
    assert r.category in CATEGORIES
    assert r.sentiment in SENTIMENT_CLASSES
    assert r.priority in PRIORITY_LEVELS
    assert r.routed_to in ALL_TEAMS


def test_predict_assigns_unique_ticket_ids(pipeline: InferencePipeline) -> None:
    ids = {pipeline.predict("hi").ticket_id for _ in range(20)}
    # 20 random 8-hex-char ids should be effectively unique.
    assert len(ids) == 20


def test_predict_respects_injected_ticket_id(pipeline: InferencePipeline) -> None:
    r = pipeline.predict("hello", ticket_id="tkt_fixed01")
    assert r.ticket_id == "tkt_fixed01"


def test_predict_routes_billing_to_billing_team(pipeline: InferencePipeline) -> None:
    r = pipeline.predict("Please refund my last invoice, the payment was duplicated.")
    assert r.category == "Billing"
    assert r.routed_to == "billing-team"


def test_predict_routes_auth_to_identity_team(pipeline: InferencePipeline) -> None:
    r = pipeline.predict("I can't log in to my account, password reset isn't working.")
    assert r.category == "Authentication"
    assert r.routed_to == "identity-team"


def test_predict_collects_urgency_signals(pipeline: InferencePipeline) -> None:
    r = pipeline.predict(
        "I was charged twice and this is unacceptable, I will sue."
    )
    assert "charged twice" in r.urgency_signals
    assert "unacceptable" in r.urgency_signals


def test_predict_reports_non_negative_latency(pipeline: InferencePipeline) -> None:
    r = pipeline.predict("hello world")
    assert isinstance(r.latency_ms, int)
    assert r.latency_ms >= 0


def test_predict_score_is_in_unit_range(pipeline: InferencePipeline) -> None:
    r = pipeline.predict("Crashed again, this is unacceptable.")
    assert 0 <= r.priority_score <= 100


def test_predict_breakdown_is_complete(pipeline: InferencePipeline) -> None:
    r = pipeline.predict("I was charged twice for my subscription!")
    assert set(r.priority_breakdown) == {"urgency", "sentiment", "plan", "confidence"}
    for value in r.priority_breakdown.values():
        assert 0.0 <= value <= 1.0


def test_predict_customer_plan_affects_score(pipeline: InferencePipeline) -> None:
    text = "I was charged twice and this is completely unacceptable, I will sue."
    r_free = pipeline.predict(text, customer_plan="free")
    r_ent = pipeline.predict(text, customer_plan="enterprise")
    # Enterprise plan should score at least as high as free, with everything
    # else held equal.
    assert r_ent.priority_score >= r_free.priority_score
    assert r_ent.priority_breakdown["plan"] > r_free.priority_breakdown["plan"]


def test_predict_rejects_non_string_text(pipeline: InferencePipeline) -> None:
    with pytest.raises(TypeError):
        pipeline.predict(123)  # type: ignore[arg-type]


def test_predict_passes_customer_id_through(pipeline: InferencePipeline) -> None:
    r = pipeline.predict("hi", customer_id="cus_xyz")
    assert r.customer_id == "cus_xyz"


def test_predict_batch_runs_each_item(pipeline: InferencePipeline) -> None:
    items = [
        ("Please refund my last invoice.", "free"),
        ("Crashed again.", "enterprise", "cus_42"),
    ]
    out = pipeline.predict_batch(items)
    assert len(out) == 2
    assert out[0].customer_id is None
    assert out[1].customer_id == "cus_42"
    assert len({r.ticket_id for r in out}) == 2  # unique ids


def test_classify_request_handles_pydantic_payload(
    pipeline: InferencePipeline,
) -> None:
    from ticket_router.schemas import ClassifyRequest

    req = ClassifyRequest(
        text="I was charged twice for my subscription!",
        customer_plan="pro",
        customer_id="cus_abc",
    )
    r = pipeline.classify_request(req)
    assert r.category == "Billing"
    assert r.customer_plan == "pro"
    assert r.customer_id == "cus_abc"


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------


def test_pipeline_without_classifier_raises_on_predict() -> None:
    p = InferencePipeline()
    with pytest.raises(RuntimeError, match="CategoryClassifier is not loaded"):
        p.predict("hello")


def test_pipeline_load_category_model_picks_up_artifact(
    tmp_path_factory: pytest.TempPathFactory,
    small_classifier: CategoryClassifier,
) -> None:
    # Persist the small classifier to a temp file and reload it through
    # the lazy loader.
    tmp = tmp_path_factory.mktemp("pipeline") / "model.joblib"
    small_classifier.save(tmp)
    p = InferencePipeline()
    assert p.category_classifier is None
    p.load_category_model(tmp)
    assert p.category_classifier is not None
    r = p.predict("Please refund my last invoice.")
    assert r.category == "Billing"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


def test_default_pipeline_is_a_singleton() -> None:
    reset_default_pipeline()
    a = get_default_pipeline()
    b = get_default_pipeline()
    assert a is b


def test_module_level_predict_uses_default_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    small_classifier: CategoryClassifier,
) -> None:
    reset_default_pipeline()
    p = get_default_pipeline()
    p.load_category_model = lambda path: None  # type: ignore[assignment]
    p.category_classifier = small_classifier

    r = predict("I was charged twice for my subscription!", customer_plan="pro")
    assert r.category == "Billing"
    assert r.routed_to == "billing-team"
    assert r.customer_plan == "pro"


def test_reset_default_pipeline_drops_cache() -> None:
    reset_default_pipeline()
    a = get_default_pipeline()
    reset_default_pipeline()
    b = get_default_pipeline()
    assert a is not b


# ---------------------------------------------------------------------------
# end-to-end with the real artifact (guarded so it skips when absent)
# ---------------------------------------------------------------------------


def test_pipeline_with_real_artifact_billing_routing() -> None:
    """Smoke test against the real trained model artifact.

    Skipped if the artifact isn't present (e.g. fresh clone without
    running ``scripts/train_category.py``). Day 12 expands this into
    the full 20+ integration test suite.
    """
    from ticket_router.config import settings

    path = settings.category_model_path
    p = InferencePipeline()
    if not p.category_classifier:
        try:
            p.load_category_model(path)
        except FileNotFoundError:
            pytest.skip(f"Category model artifact not found at {path}")

    r = p.predict(
        "I was charged twice for my subscription and this is unacceptable.",
        customer_plan="enterprise",
    )
    assert r.category == "Billing"
    assert r.routed_to == "billing-team"
    # Angry + 2 urgency signals + enterprise + high confidence should be
    # at least P2; the exact score depends on the model + analyzer state.
    assert r.priority in {"P1", "P2"}
    assert r.priority_score >= 40
    assert "charged twice" in r.urgency_signals
    assert r.sentiment in {"Angry", "Frustrated"}
