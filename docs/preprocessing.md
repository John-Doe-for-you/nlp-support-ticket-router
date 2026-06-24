# Preprocessing Analysis & Class-Weight Strategy (Day 6)

Companion to `notebooks/02_preprocessing.ipynb`. The notebook is the source of
truth for the numbers; this file is a short written summary for the repo and
for Day 7's hand-off.

---

## 1. What we did

- Rebuilt the labeled frame and the 70/15/15 stratified split from
  `data/raw/tickets_raw.csv` using `ticket_router.preprocessing.dataset`.
- Visualized the per-split class distribution and confirmed stratification is
  intact (max per-split share drift < 1pp).
- Inspected per-class text length and confirmed the median ticket is
  ~60 words (TF-IDF-friendly).
- Computed three candidate class-weight schemes and simulated the effective
  class distribution each one produces.

---

## 2. Key numbers (English rows, seed=42)

| Metric | Value |
|---|---|
| Labeled rows (English) | 24,195 |
| Train / val / test | 16,935 / 3,630 / 3,630 |
| Categories | 5 (`Billing`, `Authentication`, `Bug Report`, `Feature Request`, `Technical Setup`) |
| Counts | Bug Report 10,043; Authentication 5,415; Feature Request 3,489; Billing 3,437; Technical Setup 1,811 |
| Imbalance ratio (max/min) | **5.55x** |
| Median words per ticket | ~60 |
| P95 words per ticket | ~117 |

---

## 3. Class weights we considered

| Scheme | When to use |
|---|---|
| `none` (uniform) | Baseline only; rare classes will be under-predicted. |
| `balanced` (sklearn default) | Aggressive up-weighting of rare classes; first choice. |
| `capped_at_4` | Same as `balanced` but with an upper bound to limit noise from very small classes. |
| `effective_n` (Cui et al., 2019) | Soft inverse-frequency; smooth but the `beta=0.99` variant collapses to ~uniform on this data. |

The full weight table is in `artifacts/class_weight_strategy.json`.

---

## 4. Decision (locked in for Day 7)

Use `LogisticRegression(class_weight='balanced', solver='liblinear',
max_iter=1000, random_state=42)` as the first training run.

**Rationale**

- 5.5x imbalance is the textbook "moderate" zone where sklearn's
  inverse-frequency weight reliably improves macro-F1 without tanking
  majority-class precision.
- It is a single flag, keeping Day 7's `category_classifier.py` clean.
- `capped_at_4` and `effective_n` variants stay ready in
  `artifacts/class_weight_strategy.json` as fallbacks for Day 8.

---

## 5. Hand-off to Day 7

- Build `src/ticket_router/models/category_classifier.py` with the LogReg
  config from `artifacts/class_weight_strategy.json`.
- Train via `scripts/train_category.py` and save artifacts to
  `artifacts/category_model.joblib`, `tfidf_vectorizer.joblib`,
  `label_encoder.joblib` (gitignored, regenerated on each run).
- If Day 8 evaluation shows `Bug Report` macro-precision dropping > 5pp,
  swap to `capped_at_4` and re-train.

---

## 6. Risks logged

- `Technical Setup` has only 1.8k rows. If recall < 0.6, increase
  `ngram_range` upper bound or augment with `data/synthetic/edge_cases.jsonl`
  on Day 18.
- The whole dataset is English-only. For multi-lingual support we'd need a
  different vectorizer; out of scope for v1.