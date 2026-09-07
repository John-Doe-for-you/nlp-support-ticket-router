# NLP Support Ticket Router

![Banner](https://github.com/John-Doe-for-you/nlp-support-ticket-router/raw/main/docs/logo.png?raw=true)

> **Production-grade NLP system that automatically classifies incoming customer
> support tickets, detects sentiment, assigns a priority score, and routes the
> ticket to the correct team — all returned via a REST API in under 100ms.**

## Problem

Customer support teams receive hundreds of tickets daily across diverse categories
(Billing, Authentication, Bug Report, Feature Request, Technical Setup). Manual
routing is slow, inconsistent, and prone to human error. An automated NLP pipeline
can classify tickets in real-time, prioritize based on urgency, and route them to
the right team — improving response times and customer satisfaction.

## Architecture

```mermaid
graph TD
    A[Client] -->|POST /classify| B(FastAPI)
    B --> C[InferencePipeline]
    C --> D[Text Cleaner]
    D --> E[TF-IDF Vectorizer]
    E --> F[Category Classifier (LogReg)]
    F --> G[Category + Confidence]
    C --> H[VADER Sentiment + Custom Lexicon]
    H --> I[Sentiment + Scores]
    G & I --> J[Priority Engine]
    J --> K[Priority Score + P1/P2/P3]
    K --> L[Team Router]
    L --> M[routed_to Team]
    
    style B fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style C fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

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

## API Contract

### POST /classify

**Request:**

```json
{
  "text": "I've been charged twice for my subscription! This is unacceptable!",
  "customer_plan": "pro",
  "customer_id": "cus_123"
}
```

**Response (<100ms):**

```json
{
  "ticket_id": "tkt_abc123",
  "category": "Billing",
  "category_confidence": 0.94,
  "sentiment": "Angry",
  "sentiment_scores": {"neg": 0.78, "neu": 0.15, "pos": 0.07},
  "priority": "P1",
  "priority_score": 87,
  "routed_to": "billing-team",
  "urgency_signals": ["charged twice", "unacceptable"],
  "latency_ms": 23
}
```

### GET /stats

Returns aggregated counts by category, sentiment, priority, and team.

### GET /tickets

Pagination: `?limit=50&offset=0` — returns recent tickets with latest prediction.

### GET /health

Liveness + readiness probe. Returns 200 when fully ready, 503 when model or
DB is unavailable.

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

## Tech Stack

- **Language:** Python 3.11
- **API:** FastAPI + Uvicorn
- **ML:** TF-IDF + Logistic Regression (category), VADER + custom lexicon (sentiment), rule-based priority engine
- **DB:** SQLite + SQLAlchemy
- **Tests:** pytest + httpx TestClient
- **Container:** Docker (python:3.11-slim, multi-stage, non-root user)
- **Data:** HuggingFace `Tobi-Bueck/customer-support-tickets` + 30 hand-written edge cases

## Learnings

- The hybrid ML approach (TF-IDF + LogReg for category, VADER + custom lexicon for sentiment) offers a great balance of speed and accuracy for a portfolio project.
- Sarcasm remains the hardest pattern for VADER to catch — custom urgency lexicon helps compensate.
- Priority scoring works well when sentiment intensity, urgency keywords, and customer plan are fused together.
- Docker multi-stage builds keep the image size small (<200MB) while retaining all dependencies.
- Latency assertions should be baked in early (Day 17) rather than added later — it's much harder to retrofit.
- Maintaining a 70/15/15 stratified split with the public HuggingFace dataset required careful keyword mapping on Day 5, but it paid off in reliable per-class evaluation.

---

Built with ❤️ by [Ronak Patil](mailto:ronakpatil2406@gmail.com) — [GitHub](https://github.com/John-Doe-for-you)