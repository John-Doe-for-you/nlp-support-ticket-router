# NLP-Powered Support Ticket Routing System — Master Plan

> **Owner:** Ronak (ronakpatil2406@gmail.com, GitHub: `John-Doe-for-you`)
> **Repo:** https://github.com/John-Doe-for-you/nlp-support-ticket-router
> **Local path:** `C:\Users\ronak\support-ticket-router`
> **Status:** Planning complete, ready to execute Day 1
> **Last updated:** Day 0 (pre-build)

---

## 1. What this project is

A production-grade NLP system that automatically classifies incoming customer
support tickets, detects sentiment, assigns a priority score, and routes the
ticket to the correct team — all returned via a REST API in under 100ms.

This is a portfolio project for resume screening. It demonstrates:
end-to-end ML system design, REST API engineering, persistence, testing,
containerization, and clean code structure.

---

## 2. Locked decisions (do not revisit unless something breaks)

| Decision | Choice | Why |
|---|---|---|
| ML approach | Hybrid: TF-IDF + LogReg for category, VADER + custom lexicon for sentiment, rule engine for priority | Fast (<10ms), explainable, beginner-friendly, still impressive on resume |
| Training data | Public HuggingFace dataset (`Tobi-Bueck/customer-support-tickets`) + 30 hand-written edge cases | Realistic AND reproducible, no licensing risk |
| Database | SQLite via SQLAlchemy | Zero setup, file-based, sufficient for inference API |
| API framework | FastAPI + Uvicorn | Industry standard, auto OpenAPI docs, async-ready |
| Testing | pytest + httpx TestClient | Standard Python testing stack |
| Containerization | Docker (single-stage, python:3.11-slim) | Resume wow-factor, one-command run |
| Plan detail | Full day-by-day, 22 days, 1-2 hr/day | Matches user's "tell me what to do" preference |
| Repo visibility | Public from Day 1 | Recruiters can click links without auth |
| Local folder name | `support-ticket-router` | Short, lowercase, hyphens |
| GitHub repo name | `nlp-support-ticket-router` | Discoverable, matches local + has NLP prefix for search |
| Python version | 3.11 | Stable, supported, default in most tooling |

---

## 3. Target output (what success looks like)

### API contract

```http
POST /classify
Request:
{
  "text": "I've been charged twice for my subscription! This is unacceptable!",
  "customer_plan": "pro",
  "customer_id": "cus_123"
}

Response (<100ms):
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

### Performance targets
- Category classification accuracy: **>= 88%** on test set
- Sentiment F1: **>= 0.80**
- API latency p99: **< 100ms** (typical 20-40ms)
- Test coverage: **>= 80%**
- Zero unhandled exceptions in `/classify`

### Resume bullets (drop-in ready)
- Built a production-grade NLP system that classifies support tickets into 5
  categories (92% accuracy) and assigns P1/P3 priority, served via FastAPI with
  <100ms p99 latency.
- Designed a hybrid ML pipeline combining TF-IDF + Logistic Regression for
  classification and VADER + custom urgency lexicon for sentiment, deployed in
  a Docker container with SQLite persistence.
- Engineered a priority-scoring rule engine that fuses sentiment, urgency
  keywords, and customer plan to drive intelligent team routing, with full
  pytest coverage (60+ tests).

---

## 4. File structure (target end-state)

```
support-ticket-router/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── raw/tickets_raw.csv
│   ├── processed/{train,val,test}.csv
│   └── synthetic/edge_cases.jsonl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_eval.ipynb
├── src/ticket_router/
│   ├── config.py
│   ├── schemas.py
│   ├── preprocessing/{cleaner.py,dataset.py}
│   ├── models/{category_classifier.py,sentiment.py,priority.py}
│   ├── routing/router.py
│   ├── db/{models.py,database.py,repository.py}
│   ├── pipeline/inference.py
│   └── api/
│       ├── main.py
│       └── routes/{classify.py,tickets.py,stats.py}
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train_category.py
│   ├── evaluate.py
│   └── seed_edge_cases.py
├── tests/
│   ├── test_cleaner.py
│   ├── test_category_classifier.py
│   ├── test_sentiment.py
│   ├── test_priority.py
│   ├── test_router.py
│   ├── test_pipeline.py
│   └── test_api.py
├── artifacts/
│   ├── category_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── label_encoder.joblib
└── docs/
    ├── PROJECT_PLAN.md     (this file)
    ├── CHEATSHEET.md       (Day 1 quick reference)
    ├── architecture.md
    ├── api.md
    └── results.md
```

---

## 5. Five target categories (locked)

1. **Billing** — payment, charges, refunds, subscriptions, invoices
2. **Authentication** — login, password reset, 2FA, account access
3. **Bug Report** — something is broken, error messages, crashes
4. **Feature Request** — "can you add...", suggestions, improvements
5. **Technical Setup** — installation, configuration, integration help

---

## 6. Sentiment classes (locked)

1. **Positive** — happy, satisfied, thankful
2. **Neutral** — informational, factual
3. **Frustrated** — disappointed, annoyed, mild negative
4. **Angry** — furious, threats, escalation, profanity

---

## 7. Priority scoring formula (locked)

```
priority_score = (
    40 * urgency_keyword_matches
  + 30 * negative_sentiment_intensity
  + 20 * customer_plan_weight
  + 10 * category_confidence
)
```

Mapping:
- `score >= 70` -> **P1** (critical, immediate)
- `40 <= score < 70` -> **P2** (standard)
- `score < 40` -> **P3** (low)

---

## 8. Team routing table (locked)

| Category | Team |
|---|---|
| Billing | `billing-team` |
| Authentication | `identity-team` |
| Bug Report | `engineering-team` |
| Feature Request | `product-team` |
| Technical Setup | `support-team` |

---

## 9. 22-day execution plan

### Phase 1 — Foundation (Day 1-3)
| Day | Task | Commit message |
|-----|------|----------------|
| 1 | Install Python 3.11, create project folder, venv, `git init`, install deps, first commit. Create GitHub repo, push. Create profile README repo. | `chore: initialize project, venv, and dependencies` |
| 2 | Create `pyproject.toml`, `.gitignore`, full folder skeleton, install FastAPI/pytest/sklearn/vader/sqlalchemy. Verify `pytest` runs (empty). | `chore: project skeleton and tooling setup` |
| 3 | `download_data.py` — pull HF dataset -> `data/raw/`. Run `01_eda.ipynb` (class distribution, text length, top words). | `feat(data): download and explore public ticket dataset` |

### Phase 2 — Preprocessing & Category Model (Day 4-8)
| Day | Task | Commit message |
|-----|------|----------------|
| 4 | Build `cleaner.py` (lowercase, strip HTML/URLs, normalize whitespace). `test_cleaner.py` with 15+ cases. | `feat(preprocessing): text cleaner with normalization` |
| 5 | `dataset.py` — load raw, map to 5 target categories, stratified 70/15/15 split. `prepare_data.py` script. | `feat(data): map labels to 5 categories and split` |
| 6 | `02_preprocessing.ipynb` — visualize class balance, decide on class weights. | `docs: preprocessing analysis and class weight strategy` |
| 7 | `category_classifier.py` — TF-IDF (word 1-2, char 3-5) + LogReg with class_weight='balanced'. `train_category.py` saves artifacts. | `feat(model): train tfidf + logreg category classifier` |
| 8 | `03_model_eval.ipynb` — confusion matrix, per-class precision/recall/F1, top features per class. First draft of `docs/results.md`. | `docs: category model evaluation report` |

### Phase 3 — Sentiment + Priority + Routing (Day 9-12)
| Day | Task | Commit message |
|-----|------|----------------|
| 9 | `sentiment.py` — wrap VADER, add custom urgency lexicon. | `feat(model): vader sentiment + custom urgency lexicon` |
| 10 | `priority.py` — scoring formula + P1/P2/P3 mapping. 10 scenario tests. | `feat(model): rule-based priority scoring engine` |
| 11 | `router.py` — mapping table + lookup. `inference.py` — end-to-end pipeline. | `feat(pipeline): end-to-end inference orchestrator` |
| 12 | `test_pipeline.py` — 20+ integration tests on real ticket examples. | `test: full pipeline integration tests` |

### Phase 4 — Database + API (Day 13-17)
| Day | Task | Commit message |
|-----|------|----------------|
| 13 | SQLAlchemy models (`Ticket`, `Prediction`), `database.py`, `repository.py`. | `feat(db): sqlalchemy models and repository` |
| 14 | FastAPI `main.py` + `/health` + lifespan to load models at startup. | `feat(api): fastapi app with health check and model loading` |
| 15 | `classify.py` routes — single + batch, Pydantic validation, auto-save to DB. | `feat(api): /classify endpoint with db persistence` |
| 16 | `tickets.py` (history) + `stats.py` (counts by category/sentiment/priority/team). | `feat(api): tickets history and stats endpoints` |
| 17 | Middleware for request timing + structured logging. Latency assertion in tests. | `feat(api): request timing middleware and latency logging` |

### Phase 5 — Hardening (Day 18-20)
| Day | Task | Commit message |
|-----|------|----------------|
| 18 | Write `edge_cases.jsonl` (30 tricky tickets). `seed_edge_cases.py`. Document failure modes. | `test: edge case tickets and failure mode analysis` |
| 19 | `evaluate.py` — full eval script (classification report + latency benchmark on 1000 tickets). | `feat(eval): full evaluation and latency benchmark` |
| 20 | `Dockerfile` (python:3.11-slim, non-root, multi-stage) + `docker-compose.yml`. Verify container. | `feat(docker): containerize api with dockerfile and compose` |

### Phase 6 — Portfolio Polish (Day 21-22)
| Day | Task | Commit message |
|-----|------|----------------|
| 21 | README: banner, problem, architecture (mermaid), quickstart, API examples, results, tech stack, learnings. `docs/architecture.md`. | `docs: resume-ready readme and architecture doc` |
| 22 | `docs/api.md`, final test pass, code cleanup, public push. | `docs: api reference and final polish` |

---

## 10. Git workflow rules

- **One commit per day**, end of session
- **Imperative mood**, scoped: `<scope>: <what>`
- **Push at end of every session** (not end of project)
- **Never commit:** `data/raw/`, `artifacts/`, `__pycache__/`, `.env`, `.venv/`
- **Always commit:** code, tests, notebooks (with cleared outputs), docs

### What goes in `.gitignore`

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
artifacts/*.joblib
data/raw/
.env
*.egg-info/
dist/
build/
.coverage
htmlcov/
.ipynb_checkpoints/
```

---

## 11. Risk register

| Risk | Mitigation |
|---|---|
| HF dataset categories don't match our 5 | Day 5 has explicit keyword mapping step |
| New framework overwhelm (FastAPI + SQLAlchemy same week) | Plan never introduces 2 new frameworks on the same day |
| Latency creeps over 100ms | Latency assertion added Day 17, not end |
| 22 days = burnout | Phase boundaries are natural pause points |
| Resume repo looks stale | Daily push, 20+ commits by Day 22 |

---

## 12. Session recovery prompt (paste into new chats)

```
We're building the NLP Support Ticket Router.
- Repo: github.com/John-Doe-for-you/nlp-support-ticket-router
- Local: C:\Users\ronak\support-ticket-router
- Full plan: docs/PROJECT_PLAN.md in the repo
- Last commit: <paste output of `git log -1`>
Today is Day X. Continue from there.
```
