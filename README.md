# NLP Support Ticket Router

> Production-grade NLP system that classifies support tickets, detects sentiment,
> assigns priority, and routes to the right team — all via a REST API in <100ms.

> **Status:** Day 1 — project initialization in progress.
> See `docs/PROJECT_PLAN.md` for the full 22-day plan and `docs/CHEATSHEET.md` for Day 1 commands.

## Stack
- **Language:** Python 3.11
- **API:** FastAPI + Uvicorn
- **ML:** TF-IDF + Logistic Regression (category), VADER + custom lexicon (sentiment), rule engine (priority)
- **DB:** SQLite + SQLAlchemy
- **Tests:** pytest + httpx
- **Container:** Docker

## Quickstart (Day 1)
```powershell
# See docs/CHEATSHEET.md for the full Day 1 walkthrough
```

## Endpoints (planned)
- `POST /classify` — classify a single ticket
- `POST /classify/batch` — classify up to 100 tickets
- `GET /tickets` — list recent classifications
- `GET /stats` — dashboard aggregations
- `GET /health` — health check

## Author
Ronak Patil — [ronakpatil2406@gmail.com](mailto:ronakpatil2406@gmail.com) — [github.com/John-Doe-for-you](https://github.com/John-Doe-for-you)
