FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir pip upgrade

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ .

ENV PYTHONPATH="/app"

USER appuser

EXPOSE 8000

CMD ["uvicorn", "ticket_router.api.main:app", "--host", "0.0.0.0", "--port", "8000"]