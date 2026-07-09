# Day 14 — Summary Report

**Date:** 2026-07-09
**Owner:** Ronak
**Plan task:** FastAPI `main.py` + `/health` + lifespan to load models at startup.
**Commit:** `feat(api): fastapi app with health check and model loading`

---

## Goal of the day

Per `docs/PROJECT_PLAN.md`:

> Day 14 | FastAPI `main.py` + `/health` + lifespan to load models at startup. | `feat(api): fastapi app with health check and model loading`

So the deliverable is a runnable FastAPI app with:

* an `app` factory that builds a fresh, isolated instance per call
  (test-friendly) and a module-level singleton for `uvicorn`
* a `lifespan` context manager that warms the inference pipeline +
  the SQLAlchemy schema on startup and logs the boot state
* `GET /` (banner) and `GET /health` (liveness + readiness, 200/503)
* a test file covering all of the above

---

## What I did

1. **Pulled latest and re-aligned.** `git log -1` showed
   `50e2270 feat(db): sqlalchemy models and repository`, matching the
   session recovery prompt. Re-read `docs/PROJECT_PLAN.md` §6
   (Phase 4) and the existing modules — `pipeline.inference`,
   `db.database`, `db.models`, `config.Settings` — to design the
   app around what was already there rather than reinvent it.

2. **Designed the app factory** (`src/ticket_router/api/main.py`).
   Two entry points:
   * `create_app() -> FastAPI` for tests and programmatic use.
   * `app: FastAPI = create_app()` at module scope for
     `uvicorn ticket_router.api.main:app`.

   The factory does **not** import the Day 15/16 route stubs
   (still one-liner placeholders); mounting them is the next two
   days' job.

3. **Implemented the `lifespan`.** It:
   * logs env / model path / DB URL on startup
   * builds a fresh `InferencePipeline` and stores it on
     `app.state.pipeline` (so Day 15/16 route handlers can grab it
     without module-level globals)
   * loads the category model from `settings.category_model_path`
     via `_load_category_model()` — a failure-tolerant helper that
     logs but never raises, so the API can boot into a "degraded"
     state and `/health` can report it
   * calls `init_db(url=settings.database_url)` so the schema is
     always there before the first request. Passing the URL
     explicitly avoids the cached-engine pitfall in tests.
   * logs a shutdown line for completeness

4. **Implemented the endpoints.**
   * `GET /` (excluded from OpenAPI): banner with name, version,
     pointer to `/docs`, `/openapi.json`, `/health`.
   * `GET /health`: returns
     `{"status", "version", "model_loaded", "db_ready", "model_path",
     "database_url"}`. HTTP 200 only when **both** `model_loaded`
     and `db_ready` are true; otherwise 503. The DB URL is
     password-redacted via `_safe_db_url()`.

5. **Added CORS** wide-open by default. This is a local-dev /
   portfolio project; production deployments behind a real proxy
   should tighten `allow_origins`.

6. **Wrote `tests/test_api.py`** (16 tests):
   * `/` returns the banner and is excluded from the OpenAPI schema
   * `/health` returns 200/ok when the artifact is present
   * `/health` returns 503/degraded when the artifact is missing
   * `/health` returns 503/degraded when the artifact is corrupted
     (catches the "bad joblib" case)
   * the lifespan actually loads the model into `app.state.pipeline`
     and the loaded classifier exposes the five locked categories
   * the lifespan creates a real SQLite file at the configured path
   * the lifespan swallows model-load errors instead of crashing
   * the OpenAPI schema has the right title/version and includes
     `/health`
   * Swagger UI loads at `/docs`
   * `_safe_db_url()` redacts passwords (parametrized)
   * `create_app()` returns a fresh instance per call (test isolation)
   * the module-level `app` singleton is a real `FastAPI` instance
   * an autouse fixture scrubs `TICKET_ROUTER_*` / `DATABASE_URL` /
     `CATEGORY_MODEL_PATH` env vars so test ordering can't leak
     config

   Tests use a `monkeypatch`-set temp DB path so the checked-out
   `tickets.db` is never touched, and they share a single skip
   decorator when the real artifact is missing (CI without
   `artifacts/` is still green).

---

## What changed

| File | Status | Notes |
|---|---|---|
| `src/ticket_router/api/main.py` | replaced | 220 lines, app factory + lifespan + `/` + `/health` + `_safe_db_url` |
| `tests/test_api.py` | replaced | 16 tests covering the new surface |
| `docs/day14_summary.md` | new | this file |

No changes to `requirements.txt` (FastAPI / uvicorn / httpx were
already pinned from Day 2).

---

## Test results

```
tests/test_api.py ................                                       [  3%]
... (all 463 tests) ...
================= 463 passed, 1 warning in 105.98s (0:01:45) =================
```

The single warning is the upstream
`StarletteDeprecationWarning: Using httpx with starlette.testclient
is deprecated; install httpx2 instead` — coming from FastAPI's
vendored `TestClient` import, not our code. Tracked for the
Day 20 Docker / hardening pass.

---

## Design notes for future days

* `app.state.pipeline` is the canonical place to read the trained
  pipeline from a request handler. Day 15's `classify.py` route
  should reach for `request.app.state.pipeline`, not the module-level
  `get_default_pipeline()` singleton.
* `app.state.db_ready` is the right gate for any DB-touching
  endpoint. Day 15/16 should keep it in mind for a clean 503
  response if the DB is ever down.
* `_safe_db_url` is intentionally conservative — it only redacts
  the userinfo section. If a future deployment puts the password
  elsewhere (e.g. a `?password=` query string), extend the helper.
* CORS is wide-open for the portfolio demo. Document the change in
  `docs/architecture.md` on Day 21.

---

## Open items / known gaps

* `/metrics` (Prometheus) is not in scope until a later phase.
* No graceful-shutdown hook for in-flight requests beyond the
  default Starlette behavior. Day 17's latency middleware may want
  to add one.
* `app.state` is not documented in OpenAPI; nothing is read from it
  by the auto-generated client. If a generated client ever needs to
  see readiness, expose a separate `/ready` endpoint or move the
  state into response headers.
