# Model Evaluation Results

> Draft generated on Day 8. Final numbers filled in after Day 19's full benchmark.

## Headline numbers

- Validation accuracy: **0.882**  (macro F1 = 0.833)
- Test accuracy:       **0.889**  (macro F1 = 0.840)
- Avg model latency: **0.470 ms/ticket**

Target thresholds (from `docs/PROJECT_PLAN.md`): accuracy >= 0.88, sentiment F1 >= 0.80, API p99 < 100 ms.

## Per-class metrics (test split)

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Billing | 0.915 | 0.754 | 0.827 | 516 |
| Authentication | 0.890 | 0.943 | 0.916 | 812 |
| Bug Report | 0.932 | 0.962 | 0.947 | 1507 |
| Feature Request | 0.820 | 0.864 | 0.842 | 523 |
| Technical Setup | 0.717 | 0.625 | 0.668 | 272 |

## Top features per class

- **Authentication** — `login`, `security`, `breach`, `medical`, `access`, `data breach`, `hospital`, `login`
- **Billing** — `billing`, `payment`, `pricing`, `the billing`, `bill`, ` bill`, ` bil`, `promote`
- **Bug Report** — `performance`, `outage`, `outages`, `crashes`, `service`, `problem`, `crash`, `downtime`
- **Feature Request** — `support for`, `enhancement`, `integrate`, `features`, `integration`, `enhancements`, `to integrate`, `integration with`
- **Technical Setup** — `how to`, `documentation`, `how`, `on how`, `configuration`, `guidelines`, `network configuration`, `setup`

## Failure modes observed

(See notebook section 5 for the live table. Day 18 will expand this with the edge-case suite.)

