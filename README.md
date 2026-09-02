# NLP Support Ticket Router

> Production-grade NLP system that classifies support tickets, detects sentiment,
> assigns priority, and routes to the right team — all via a REST API in <100ms.

## Status

Completed Days 1-20: Full pipeline implemented and containerized. API served via FastAPI with SQLite persistence, model training, evaluation, and Docker deployment.

## Stack

- **Language:** Python 3.11
- **API:** FastAPI + Uvicorn
- **ML:** TF-IDF + Logistic Regression (category), VADER + custom lexicon (sentiment), rule-based priority engine
- **DB:** SQLite + SQLAlchemy
- **Tests:** pytest + httpx TestClient
- **Container:** Docker (python:3.11-slim, multi-stage, non-root user)

## Features

- Ticket classification into 5 categories (Billing, Authentication, Bug Report, Feature Request, Technical Setup)
- Sentiment analysis (Positive, Neutral, Frustrated, Angry) using VADER + custom urgency lexicon
- Priority scoring (P1/P2/P3) based on sentiment, urgency keywords, customer plan, and category confidence
- Team routing to 5 specialized teams
- REST API with `/classify` (single + batch), `/tickets` (history), `/stats` (aggregations), `/health` (probe)
- SQLite persistence with SQLAlchemy repository pattern
- Full test coverage with pytest + httpx
- Docker containerization with production-ready image

## API Contract

```http
POST /classify
{
  "text": "...",
  "customer_plan": "pro",
  "customer_id": "cus_123"
}
```

Response includes: `ticket_id`, `category`, `category_confidence`, `sentiment`, `sentiment_scores`, `priority`, `priority_score`, `routed_to`, `urgency_signals`, `latency_ms`

## Run Locally

```powershell
docker compose up --build
```

Or locally:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn ticket_router.api.main:app --reload
```