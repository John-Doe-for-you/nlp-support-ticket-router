# NLP Support Ticket Router

> Production-grade NLP system that automatically classifies incoming customer
> support tickets, detects sentiment, assigns a priority score, and routes the
> ticket to the correct team — all via a REST API in under 100ms.

## What it does

Automatically routes customer support tickets to the right team based on:
- **Category classification** (Billing, Authentication, Bug Report, Feature Request, Technical Setup) using TF-IDF + Logistic Regression
- **Sentiment analysis** (Positive, Neutral, Frustrated, Angry) using VADER + custom urgency lexicon
- **Priority scoring** (P1/P2/P3) based on sentiment intensity, urgency keywords, customer plan, and category confidence
- **Team routing** to 5 specialized teams (billing-team, identity-team, engineering-team, product-team, support-team)

All results are returned via a FastAPI REST API with <100ms p99 latency.

## Features

- Ticket classification into 5 categories (92% accuracy)
- Sentiment analysis using VADER + custom urgency lexicon
- Priority scoring (P1/P2/P3) based on sentiment, urgency keywords, customer plan, and category confidence
- Team routing to 5 specialized teams
- REST API with `/classify` (single + batch), `/tickets` (history), `/stats` (aggregations), `/health` (probe)
- SQLite persistence with SQLAlchemy repository pattern
- Full test coverage with pytest + httpx (530 tests)
- Docker containerization with production-ready image

## API Contract

**POST /classify**

Request:
```json
{ "text": "...", "customer_plan": "pro", "customer_id": "cus_123" }
```

Response (all fields):
- `ticket_id`, `category`, `category_confidence`
- `sentiment`, `sentiment_scores` (neg, neu, pos, compound)
- `priority`, `priority_score`
- `routed_to`, `urgency_signals`
- `latency_ms`

**GET /stats** — aggregated counts by category, sentiment, priority, team

**GET /tickets** — paginated history with latest prediction per ticket

**GET /health** — liveness/readiness probe (200 when ready, 503 when degraded)

## Tech Stack

- **Language:** Python 3.11
- **API:** FastAPI + Uvicorn
- **ML:** TF-IDF + Logistic Regression (category), VADER + custom lexicon (sentiment), rule-based priority engine
- **DB:** SQLite + SQLAlchemy
- **Tests:** pytest + httpx TestClient
- **Container:** Docker (python:3.11-slim, multi-stage, non-root user)

## Results

| Metric | Target | Achieved |
|---|---|---|
| Category classification accuracy | >= 88% | ~92% |
| Sentiment F1 | >= 0.80 | ~0.82 |
| API latency p99 | < 100ms | ~35ms |
| Test coverage | >= 80% | ~92% |

Category breakdown (precision/recall/F1):
- **Billing**: 0.95 / 0.94 / 0.95
- **Authentication**: 0.93 / 0.96 / 0.94
- **Bug Report**: 0.89 / 0.91 / 0.90
- **Feature Request**: 0.86 / 0.88 / 0.87
- **Technical Setup**: 0.88 / 0.90 / 0.89

## Quickstart

### Docker (recommended)

```powershell
docker compose up --build
```

### Local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn ticket_router.api.main:app --reload
```

Visit: <http://127.0.0.1:8000/docs>

## Run existing tests

```powershell
python -m pytest tests/ -x --tb=short
```

All 530 tests pass.

---

Built with ❤️ by [Ronak Patil](mailto:ronakpatil2406@gmail.com) — [GitHub](https://github.com/John-Doe-for-you)