# Day 7 — Summary Report

**Date:** 2026-06-25
**Owner:** Ronak
**Plan task:** `category_classifier.py` — TF-IDF (word 1-2, char 3-5) + LogReg with `class_weight='balanced'`. `train_category.py` saves artifacts.
**Commit:** `feat(model): train tfidf + logreg category classifier`

---

## Goal of the day

Per `docs/PROJECT_PLAN.md`:

> Day 7 | `category_classifier.py` — TF-IDF (word 1-2, char 3-5) + LogReg with `class_weight='balanced'`. `train_category.py` saves artifacts. | `feat(model): train tfidf + logreg category classifier`

So the deliverable is a working first-cut category model, with artifacts on disk
and tests that lock in behaviour.

---

## What I did

1. **Pulled latest and re-aligned.** `git log -1` showed `de8aeeb docs: day 6
   summary and error reports`, matching the session recovery prompt. Re-read
   the plan and the Day 6 hand-off notes (`docs/day6_summary.md`,
   `docs/day6_errors.md`) for the locked config:
   `LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000, random_state=42)`.

2. **Picked up where collaborator left off.** The Day 7 files
   (`src/ticket_router/models/category_classifier.py`,
   `scripts/train_category.py`, `tests/test_category_classifier.py`) were
   already written but **uncommitted** in the working tree. I read each one
   end-to-end to verify the implementation matched the plan and Day 6
   decisions, then ran the tests before changing anything. All 20 new tests
   passed on the first run; no edits needed.

3. **Ran the training script** against the prepared splits in
   `data/processed/{train,val}.csv` (16,935 / 3,630 rows). Fit time: ~65 s on
   the local venv. Inference time: **~1.3 ms per ticket** (300-ticket
   batch = 379 ms), well under the locked <10 ms target.

4. **Saved the model artifact** to `artifacts/category_model.joblib`
   (5.6 MB, gitignored) and a `train_metrics.json` summary to
   `artifacts/train_metrics.json` (tracked, 5 KB, documents this commit's
   model performance).

5. **Ran the full fast test suite** (64 tests across `test_cleaner.py`,
   `test_dataset.py`, `test_category_classifier.py`) — 64/64 green.
   Placeholder files for future days (`test_api.py`, `test_sentiment.py`,
   etc.) contain only docstrings; they are intentionally empty and skipped
   by pytest today.

6. **Wrote this report and the companion error report** for the other
   contributor.

---

## Key numbers from the trained model

Run on the held-out **val** split (3,630 rows):

| Metric | Value | Target | Status |
|---|---|---|---|
| Val accuracy | **0.8818** | >= 0.88 | ✅ just over |
| Val macro F1 | **0.8331** | >= 0.80 | ✅ comfortable |
| Val weighted F1 | 0.8817 | — | — |
| Train accuracy | 0.9251 | — | (mild overfit) |

**Per-class val metrics:**

| Category | P | R | F1 | support |
|---|---|---|---|---|
| Bug Report | 0.925 | 0.951 | 0.938 | 1507 |
| Authentication | 0.889 | 0.938 | 0.913 | 812 |
| Billing | 0.925 | 0.766 | 0.838 | 516 |
| Feature Request | 0.798 | 0.847 | 0.822 | 523 |
| Technical Setup | 0.697 | 0.618 | 0.655 | 272 |

Observations:

- The model passes both locked targets, but just barely on accuracy (0.8818
  vs 0.88). Day 8's eval notebook should treat accuracy as something to
  defend / improve.
- **`Technical Setup` is the weakest class** by a wide margin
  (F1 = 0.655, support = 272). This is the smallest class, the same one
  Day 6's imbalance analysis flagged as most at risk. It is also the most
  semantically fuzzy class (any "how to install / configure / integrate"
  ticket that isn't a clear bug), so we expect confusion with `Bug Report`
  and `Feature Request`.
- **`Billing` has the lowest recall** (0.766) — when the model is wrong
  about a Billing ticket, it tends to predict something else. Worth
  digging into the confusion matrix in Day 8.
- Train-val gap is ~4 pp on accuracy and ~7 pp on macro F1: expected for
  TF-IDF + linear model with `class_weight='balanced'` boosting the small
  classes on train. Not a red flag.

---

## What's in this commit

| Status | Path | Notes |
|---|---|---|
| Modified | `src/ticket_router/models/category_classifier.py` | Pipeline + `CategoryClassifier` wrapper. |
| Modified | `scripts/train_category.py` | Train + persist + emit metrics. |
| Modified | `tests/test_category_classifier.py` | 20 new tests (structural, cleaner integration, semantic smoke tests, save/load, on-real-data). |
| Added | `artifacts/category_model.joblib` | 5.6 MB, gitignored. |
| Added | `artifacts/train_metrics.json` | 5 KB, tracked (documents this commit's model). |
| Added | `docs/day7_summary.md` | This file. |
| Added | `docs/day7_errors.md` | Companion error report. |
| Added | `artifacts/train_run.log` | Stdout from the training run, gitignored (`*.log` rule). |

---

## How to reproduce

```bash
# Requires processed splits + raw CSV (gitignored).
.venv\Scripts\python.exe scripts\prepare_data.py
.venv\Scripts\python.exe scripts\train_category.py
.venv\Scripts\python.exe -m pytest tests\test_category_classifier.py
```

---

## Next session — Day 8

Per the plan:

- Build `notebooks/03_model_eval.ipynb` with confusion matrix, per-class
  precision/recall/F1, and top features per class.
- First draft of `docs/results.md`.
- Use the Day 6 strategy JSON to justify the `balanced` weighting choice
  in the writeup.
- Worth investigating: is the `Technical Setup` weakness caused by label
  noise in `map_to_category` (over-eager keyword matching), by genuine
  class overlap, or by simply too few training examples? A 5-10 ticket
  error analysis pass before the notebook will save time in the writeup.
