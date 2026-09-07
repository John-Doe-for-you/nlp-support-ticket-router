# API Reference

## Base URL

`http://127.0.0.1:8000`

## Authentication

No authentication required. Endpoints are open by default for local development.

## Endpoints

### POST /classify

Classify a single support ticket and return category, sentiment, priority, and routing.

**Request:**

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | `str` | Yes | The ticket text content |
| `customer_plan` | `str` | No | Customer plan: `free` or `pro`. Defaults to `free` |
| `customer_id` | `str` | No | Customer identifier |

**Request example:**

```bash
curl -X POST "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text":"I've been charged twice for my subscription!","customer_plan":"pro","customer_id":"cus_123"}'
```

**Response (200):**

| Field | Type | Description |
|---|---|---|
| `ticket_id` | `str` | Unique ticket identifier (e.g., `tkt_abc123`) |
| `category` | `str` | One of: `Billing`, `Authentication`, `Bug Report`, `Feature Request`, `Technical Setup` |
| `category_confidence` | `float` | Confidence score in [0, 1] |
| `sentiment` | `str` | One of: `Positive`, `Neutral`, `Frustrated`, `Angry` |
| `sentiment_scores` | `object` | `{"neg": float, "neu": float, "pos": float, "compound": float}` |
| `priority` | `str` | One of: `P1`, `P2`, `P3` |
| `priority_score` | `int` | Weighted score used to determine priority |
| `routed_to` | `str` | Team name (e.g., `billing-team`) |
| `urgency_signals` | `array[str]` | Keywords/phrases that triggered priority escalation |
| `latency_ms` | `int` | Handler-only processing time in milliseconds |

**Response example:**

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

---

### POST /classify/batch

Classify multiple tickets in a single request.

**Request:**

| Field | Type | Required | Description |
|---|---|---|---|
| `items` | `array[str]` | Yes | List of ticket text strings. Max 100 items. |

**Request example:**

```bash
curl -X POST "http://127.0.0.1:8000/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{"items":["I want a refund","Great service!","Login keeps failing"]}'
```

**Response (200):**

```json
{
  "count": 3,
  "latency_ms": 45,
  "results": [ ClassifyResponse, ... ]
}
```

---

### GET /stats

Return aggregated counts by category, sentiment, priority, and team.

**Response (200):**

```json
{
  "total_predictions": 128,
  "total_tickets": 120,
  "by_category": { "total": 128, "items": [{"Billing": 27}, {"Authentication": 22}, ...] },
  "by_sentiment": { "total": 128, "items": [{"Angry": 31}, {"Neutral": 28}, ...] },
  "by_priority": { "total": 128, "items": [{"P1": 12}, {"P2": 45}, {"P3": 71}] },
  "by_team": { "total": 128, "items": [{"billing-team": 27}, {"identity-team": 22}, ...] }
}
```

---

### GET /tickets

List recent tickets with pagination.

**Query Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `limit` | `int` | No | Items per page (1-200, default 50, clamped to 200) |
| `offset` | `int` | No | Items to skip from the start (default 0) |

**Response (200):**

```json
{
  "count": 50,
  "total": 128,
  "limit": 50,
  "offset": 0,
  "items": [
    {
      "ticket_id": "tkt_abc123",
      "customer_id": "cus_123",
      "customer_plan": "pro",
      "text_preview": "I want a refund for...",
      "text_length": 47,
      "created_at": "2024-01-15T10:30:00Z",
      "latest_prediction": {
        "category": "Billing",
        "sentiment": "Angry",
        "priority": "P1",
        "routed_to": "billing-team"
      }
    }
  ]
}
```

**Single ticket detail:**

`GET /tickets/{ticket_id}` — returns full ticket detail with all predictions.

---

### GET /health

Liveness and readiness probe.

**Response (200 - fully ready):**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model_loaded": true,
  "db_ready": true,
  "model_path": "artifacts/category_model.joblib",
  "database_url": "sqlite:///./tickets.db"
}
```

**Response (503 - degraded):**

```json
{
  "status": "degraded",
  "version": "0.1.0",
  "model_loaded": false,
  "db_ready": true,
  "model_path": "artifacts/category_model.joblib",
  "database_url": "sqlite:///./tickets.db"
}
```