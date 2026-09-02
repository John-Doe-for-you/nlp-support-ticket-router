# Model Evaluation & Failure Mode Analysis

## 1. Category Classification (TF-IDF + LogReg)
- **Target Accuracy:** >= 88%
- **Observations:**
    - Strong performance on "Billing" and "Authentication" due to high keyword density (e.g., "invoice", "password").
    - Confusion occasionally occurs between "Bug Report" and "Technical Setup" when the user describes an installation error as a "bug".

## 2. Sentiment Analysis (VADER + Custom Lexicon)
- **Target F1:** >= 0.80
- **Observations:**
    - Sarcasm remains a challenge (e.g., "Great, another error!").
    - Custom urgency lexicon successfully boosts priority for phrases like "unacceptable service" or "immediately".

## 3. Priority Scoring Engine
- **Formula:** `40 * urgency + 30 * sentiment + 20 * plan + 10 * confidence`
- **Edge Case Analysis:**
    - **False P1s:** Extremely angry users on free plans may hit P1 threshold despite low business impact.
    - **Missed P1s:** Calmly worded but critical system outages (e.g., "The API is returning 500s for all users") might be rated P2 if sentiment is too neutral.

## 4. Identified Failure Modes
| Scenario | Expected | Actual | Root Cause |
|---|---|---|---|
| Sarcastic complaint | P1/Angry | P3/Positive | VADER interprets "Great" as positive. |
| Technical jargon | Bug Report | Tech Setup | Overlap in "installation" vs "crash" keywords. |
| Calm critical outage | P1 | P2 | Priority score heavily weighted on sentiment intensity. |
