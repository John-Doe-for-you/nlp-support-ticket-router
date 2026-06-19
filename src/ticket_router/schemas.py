"""Pydantic request/response schemas for the API."""

from typing import Literal

from pydantic import BaseModel, Field

CustomerPlan = Literal["free", "pro", "enterprise"]
Sentiment = Literal["Positive", "Neutral", "Frustrated", "Angry"]
Priority = Literal["P1", "P2", "P3"]
Category = Literal["Billing", "Authentication", "Bug Report", "Feature Request", "Technical Setup"]


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000, description="Raw support ticket text")
    customer_plan: CustomerPlan = "free"
    customer_id: str | None = None


class SentimentScores(BaseModel):
    neg: float
    neu: float
    pos: float
    compound: float


class ClassifyResponse(BaseModel):
    ticket_id: str
    category: Category
    category_confidence: float
    sentiment: Sentiment
    sentiment_scores: SentimentScores
    priority: Priority
    priority_score: int
    routed_to: str
    urgency_signals: list[str]
    latency_ms: int
