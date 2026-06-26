"""Build 03_model_eval.ipynb programmatically with cleared outputs.

Mirrors the pattern used by `scripts/build_nb02.py` so the notebook stays
under source control without binary image blobs. Run from repo root:

    .venv/Scripts/python.exe scripts/build_nb03.py

The notebook consumes:
- `artifacts/category_model.joblib` (produced by `scripts/train_category.py`)
- `data/processed/{val,test}.csv`   (produced by `scripts/prepare_data.py`)

and writes:
- `docs/results.md`                 (draft, expanded on later days)
- `artifacts/eval_metrics.json`     (per-class metrics on val + test)
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "03_model_eval.ipynb"
ARTIFACTS = REPO / "artifacts"
PROCESSED = REPO / "data" / "processed"
RESULTS_MD = REPO / "docs" / "results.md"
EVAL_METRICS_PATH = ARTIFACTS / "eval_metrics.json"


def md(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell("\n".join(lines))


def code(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("\n".join(lines))


cells: list[nbf.NotebookNode] = []

cells.append(md(
    "# 03 \u2014 Category Model Evaluation",
    "",
    "**Goal of this notebook**",
    "",
    "1. Reload the Day-7 pipeline (`artifacts/category_model.joblib`) and re-evaluate it",
    "   on the held-out **validation** and **test** splits.",
    "2. Produce a confusion matrix and per-class precision / recall / F1 for both splits.",
    "3. Inspect the top-N n-gram features the model relies on per class (LogReg coefficients).",
    "4. Quantify **confidence calibration** so we can pick a sensible confidence threshold",
    "   for the Day-15 `/classify` endpoint.",
    "5. Write the first draft of `docs/results.md` from these numbers.",
    "",
    "> Day 8 deliverable per `docs/PROJECT_PLAN.md`. Outputs are stripped from the saved",
    "> notebook; `scripts/build_nb03.py` regenerates it deterministically."
))

# ---------------------------------------------------------------------- 0
cells.append(md("## 0. Setup"))
cells.append(code(
    "import json\n",
    "import sys\n",
    "import time\n",
    "from pathlib import Path\n",
    "\n",
    "import matplotlib\n",
    "matplotlib.use(\"Agg\")\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import (\n",
    "    classification_report,\n",
    "    confusion_matrix,\n",
    "    precision_recall_fscore_support,\n",
    ")\n",
    "\n",
    "_here = Path.cwd().resolve()\n",
    "REPO = next((p for p in (_here, *_here.parents) if (p / \"src\").is_dir()), _here)\n",
    "if str(REPO) not in sys.path:\n",
    "    sys.path.insert(0, str(REPO))\n",
    "sys.path.insert(0, str(REPO / \"src\"))\n",
    "\n",
    "from ticket_router.models.category_classifier import (\n",
    "    CATEGORIES,\n",
    "    CategoryClassifier,\n",
    ")\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\", context=\"notebook\")\n",
    "plt.rcParams[\"figure.figsize\"] = (8, 5)\n",
    "print(\"Python:\", sys.version.split()[0])\n",
    "print(\"Repo root:\", REPO)\n",
    "print(\"Target categories:\", list(CATEGORIES))"
))

# ---------------------------------------------------------------------- 1
cells.append(md(
    "## 1. Reload model + splits",
    "",
    "We always reload the artifact instead of refitting so this notebook is",
    "fully reproducible from the saved Day-7 model."
))
cells.append(code(
    "model_path = REPO / \"artifacts\" / \"category_model.joblib\"\n",
    "assert model_path.exists(), f\"Train first: python scripts/train_category.py  (missing {model_path})\"\n",
    "\n",
    "clf = CategoryClassifier.load(model_path)\n",
    "print(\"Loaded pipeline:\", clf.pipeline)\n",
    "print(\"Classes:\", clf.classes_())"
))
cells.append(code(
    "def _load(name: str) -> tuple[list[str], list[str]]:\n",
    "    df = pd.read_csv(REPO / \"data\" / \"processed\" / f\"{name}.csv\")\n",
    "    return df[\"text\"].astype(str).tolist(), df[\"category\"].astype(str).tolist()\n",
    "\n",
    "val_texts,  val_labels  = _load(\"val\")\n",
    "test_texts, test_labels = _load(\"test\")\n",
    "print(f\"val  rows: {len(val_labels):,}\")\n",
    "print(f\"test rows: {len(test_labels):,}\")"
))

# ---------------------------------------------------------------------- 2
cells.append(md("## 2. Confusion matrix + per-class metrics"))
cells.append(code(
    "def report(y_true, y_pred) -> dict:\n",
    "    labels = list(CATEGORIES)\n",
    "    p, r, f, s = precision_recall_fscore_support(\n",
    "        y_true, y_pred, labels=labels, zero_division=0\n",
    "    )\n",
    "    return {\n",
    "        \"labels\": labels,\n",
    "        \"precision\": p, \"recall\": r, \"f1\": f, \"support\": s,\n",
    "        \"report\": classification_report(y_true, y_pred, labels=labels, zero_division=0),\n",
    "        \"confusion\": confusion_matrix(y_true, y_pred, labels=labels),\n",
    "    }\n",
    "\n",
    "t0 = time.perf_counter()\n",
    "val_pred  = clf.predict(val_texts)\n",
    "test_pred = clf.predict(test_texts)\n",
    "infer_seconds = time.perf_counter() - t0\n",
    "print(f\"Inferred val+test in {infer_seconds:.2f}s \"\n",
    "      f\"({infer_seconds/(len(val_labels)+len(test_labels))*1000:.2f} ms/ticket)\")"
))
cells.append(code(
    "val_r  = report(val_labels,  val_pred)\n",
    "test_r = report(test_labels, test_pred)\n",
    "print(\"VAL classification report\\n\" + val_r[\"report\"])\n",
    "print(\"TEST classification report\\n\" + test_r[\"report\"])"
))
cells.append(code(
    "def plot_cm(cm: np.ndarray, labels: list[str], title: str) -> None:\n",
    "    fig, ax = plt.subplots(figsize=(7, 6))\n",
    "    sns.heatmap(\n",
    "        cm, annot=True, fmt=\"d\", cmap=\"Blues\",\n",
    "        xticklabels=labels, yticklabels=labels, cbar=False, ax=ax,\n",
    "    )\n",
    "    ax.set_xlabel(\"Predicted\")\n",
    "    ax.set_ylabel(\"True\")\n",
    "    ax.set_title(title)\n",
    "    plt.xticks(rotation=20)\n",
    "    plt.yticks(rotation=20)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "plot_cm(val_r[\"confusion\"],  list(CATEGORIES), \"Validation confusion matrix\")\n",
    "plot_cm(test_r[\"confusion\"], list(CATEGORIES), \"Test confusion matrix\")"
))
cells.append(code(
    "per_class = pd.DataFrame(\n",
    "    {\n",
    "        \"val_P\":  val_r[\"precision\"],\n",
    "        \"val_R\":  val_r[\"recall\"],\n",
    "        \"val_F1\": val_r[\"f1\"],\n",
    "        \"val_n\":  val_r[\"support\"].astype(int),\n",
    "        \"test_P\": test_r[\"precision\"],\n",
    "        \"test_R\": test_r[\"recall\"],\n",
    "        \"test_F1\":test_r[\"f1\"],\n",
    "        \"test_n\": test_r[\"support\"].astype(int),\n",
    "    },\n",
    "    index=list(CATEGORIES),\n",
    ")\n",
    "per_class.round(3)"
))

# ---------------------------------------------------------------------- 3
cells.append(md(
    "## 3. Top-N features per class",
    "",
    "LogReg coefficients over a TF-IDF FeatureUnion give us a faithful",
    "interpretability window. We read the coefficient matrix from the trained",
    "pipeline, rank the union of word + char n-gram features, and show the",
    "top 15 positive contributors for each class."
))
cells.append(code(
    "logreg = clf.pipeline.named_steps[\"clf\"]\n",
    "feature_names = clf.feature_names()\n",
    "coef = logreg.coef_                 # shape (n_classes, n_features)\n",
    "classes = clf.classes_()\n",
    "print(\"feature_names len:\", len(feature_names))\n",
    "print(\"coef shape:\", coef.shape, \"classes:\", classes)"
))
cells.append(code(
    "TOP = 15\n",
    "top_features: dict[str, list[tuple[str, float]]] = {}\n",
    "for i, cls in enumerate(classes):\n",
    "    idx = np.argsort(coef[i])[::-1][:TOP]\n",
    "    top_features[cls] = [(feature_names[j], float(coef[i, j])) for j in idx]\n",
    "\n",
    "for cls, feats in top_features.items():\n",
    "    print(f\"\\n[{cls}]  top {TOP} features\")\n",
    "    for name, score in feats:\n",
    "        print(f\"  {score:+8.3f}  {name}\")"
))
cells.append(code(
    "n_classes = len(classes)\n",
    "fig, axes = plt.subplots(n_classes, 1, figsize=(8, 1.7 * n_classes), sharex=False)\n",
    "for ax, cls in zip(axes, classes):\n",
    "    feats = top_features[cls][:10][::-1]  # bottom-up for horizontal bar\n",
    "    ax.barh([f for f, _ in feats], [s for _, s in feats], color=\"#3b82f6\")\n",
    "    ax.set_title(f\"Top features -> {cls}\", loc=\"left\", fontsize=11)\n",
    "    ax.tick_params(axis=\"y\", labelsize=9)\n",
    "plt.tight_layout()\n",
    "plt.show()"
))

# ---------------------------------------------------------------------- 4
cells.append(md(
    "## 4. Confidence calibration + threshold sweep",
    "",
    "Confidence is the max probability returned by `predict_proba`. We want to",
    "know:",
    "",
    "* How concentrated is the score mass on the predicted class?",
    "* At what threshold does `predicted_label == true_label` exceed 95%?",
    "* What fraction of tickets fall below each threshold? (for fallback routing)"
))
cells.append(code(
    "def confidence_block(texts, labels) -> pd.DataFrame:\n",
    "    proba = clf.predict_proba(texts)\n",
    "    pred = proba.argmax(axis=1)\n",
    "    conf = proba.max(axis=1)\n",
    "    correct = np.array([classes[p] == t for p, t in zip(pred, labels)])\n",
    "    df = pd.DataFrame({\"conf\": conf, \"correct\": correct})\n",
    "    return df\n",
    "\n",
    "val_conf  = confidence_block(val_texts,  val_labels)\n",
    "test_conf = confidence_block(test_texts, test_labels)\n",
    "val_conf.head()"
))
cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)\n",
    "for ax, df, name in [(axes[0], val_conf, \"val\"), (axes[1], test_conf, \"test\")]:\n",
    "    bins = np.linspace(0, 1, 21)\n",
    "    ax.hist(df[df.correct][\"conf\"], bins=bins, alpha=0.7, label=\"correct\", color=\"#10b981\")\n",
    "    ax.hist(df[~df.correct][\"conf\"], bins=bins, alpha=0.7, label=\"wrong\",   color=\"#ef4444\")\n",
    "    ax.set_title(f\"Confidence histogram \u2014 {name}\")\n",
    "    ax.set_xlabel(\"max(proba)\")\n",
    "    ax.set_ylabel(\"# tickets\")\n",
    "    ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
))
cells.append(code(
    "thresholds = np.round(np.arange(0.30, 0.96, 0.05), 2)\n",
    "rows = []\n",
    "for t in thresholds:\n",
    "    for df, name in [(val_conf, \"val\"), (test_conf, \"test\")]:\n",
    "        covered = df[\"conf\"] >= t\n",
    "        n_covered = int(covered.sum())\n",
    "        if n_covered == 0:\n",
    "            continue\n",
    "        acc = float(df.loc[covered, \"correct\"].mean())\n",
    "        coverage = float(n_covered / len(df))\n",
    "        rows.append({\"threshold\": t, \"split\": name, \"coverage\": coverage, \"acc_at_or_above\": acc, \"n\": n_covered})\n",
    "thr_df = pd.DataFrame(rows).round(4)\n",
    "thr_df"
))
cells.append(code(
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "for split, color in [(\"val\", \"#3b82f6\"), (\"test\", \"#f59e0b\")]:\n",
    "    sub = thr_df[thr_df[\"split\"] == split]\n",
    "    ax.plot(sub[\"threshold\"], sub[\"acc_at_or_above\"], marker=\"o\", color=color, label=f\"acc >= threshold ({split})\")\n",
    "    ax.plot(sub[\"threshold\"], sub[\"coverage\"],       marker=\"s\", color=color, linestyle=\"--\", alpha=0.5, label=f\"coverage ({split})\")\n",
    "ax.set_xlabel(\"Confidence threshold\")\n",
    "ax.set_ylabel(\"Accuracy / Coverage\")\n",
    "ax.set_title(\"Threshold sweep: accuracy-on-covered vs coverage\")\n",
    "ax.axvline(0.60, color=\"grey\", linestyle=\":\", linewidth=1)\n",
    "ax.legend(loc=\"lower left\", fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.show()"
))

# ---------------------------------------------------------------------- 5
cells.append(md(
    "## 5. Where do the mistakes go?",
    "",
    "Quick look at the most frequent (true -> predicted) mis-classifications on the",
    "test split. These are the failure modes we want to attack in Day 18's edge-case work."
))
cells.append(code(
    "pairs = pd.DataFrame({\"true\": test_labels, \"pred\": test_pred})\n",
    "wrong = pairs[pairs.true != pairs.pred]\n",
    "confusion_pairs = (\n",
    "    wrong.groupby([\"true\", \"pred\"]).size()\n",
    "    .sort_values(ascending=False)\n",
    "    .head(10)\n",
    "    .rename(\"count\")\n",
    "    .reset_index()\n",
    ")\n",
    "confusion_pairs"
))
cells.append(code(
    "test_proba = clf.predict_proba(test_texts)\n",
    "test_conf  = test_proba.max(axis=1)\n",
    "wrong_with_conf = wrong.copy()\n",
    "wrong_with_conf[\"conf\"] = test_conf[pairs.true != pairs.pred]\n",
    "wrong_with_conf.sort_values(\"conf\", ascending=False).head(10)"
))

# ---------------------------------------------------------------------- 6
cells.append(md(
    "## 6. Per-ticket latency on test (proxy for API p99)",
    "",
    "We measure the classifier's end-to-end latency on the test set. The full",
    "`/classify` request will add cleaning + DB write overhead, but the model",
    "is the dominant cost."
))
cells.append(code(
    "samples = 200 if len(test_texts) > 200 else len(test_texts)\n",
    "sample_texts = test_texts[:samples]\n",
    "timings = []\n",
    "for _ in range(3):\n",
    "    t0 = time.perf_counter()\n",
    "    clf.predict_proba(sample_texts)\n",
    "    timings.append((time.perf_counter() - t0) / samples * 1000)\n",
    "print(f\"Avg ms/ticket over 3 runs of {samples} tickets: {np.mean(timings):.3f} ms (min {min(timings):.3f}, max {max(timings):.3f})\")"
))

# ---------------------------------------------------------------------- 7
cells.append(md(
    "## 7. Persist eval metrics + draft results.md",
    "",
    "Two artifacts come out of this notebook:",
    "",
    "* `artifacts/eval_metrics.json` \u2014 numeric summary for downstream scripts.",
    "* `docs/results.md`            \u2014 human-readable first draft of the Day-8 results report."
))
cells.append(code(
    "def _pack(y_true, y_pred, proba, name):\n",
    "    labels = list(CATEGORIES)\n",
    "    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)\n",
    "    cm = confusion_matrix(y_true, y_pred, labels=labels)\n",
    "    return {\n",
    "        \"accuracy\": float((np.array(y_true) == np.array(y_pred)).mean()),\n",
    "        \"macro_f1\": float(np.mean(f)),\n",
    "        \"weighted_f1\": float(np.average(f, weights=s)),\n",
    "        \"per_class\": {\n",
    "            labels[i]: {\"precision\": float(p[i]), \"recall\": float(r[i]),\n",
    "                        \"f1\": float(f[i]), \"support\": int(s[i])}\n",
    "            for i in range(len(labels))\n",
    "        },\n",
    "        \"confusion_matrix\": cm.tolist(),\n",
    "        \"labels\": labels,\n",
    "        \"n\": int(len(y_true)),\n",
    "        \"name\": name,\n",
    "    }\n",
    "\n",
    "val_proba = clf.predict_proba(val_texts)\n",
    "test_proba = clf.predict_proba(test_texts)\n",
    "\n",
    "EVAL_METRICS_PATH = REPO / \"artifacts\" / \"eval_metrics.json\"\n",
    "RESULTS_MD = REPO / \"docs\" / \"results.md\"\n",
    "EVAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "eval_payload = {\n",
    "    \"val\":  _pack(val_labels,  val_pred,  val_proba,  \"val\"),\n",
    "    \"test\": _pack(test_labels, test_pred, test_proba, \"test\"),\n",
    "    \"latency_ms_per_ticket\": float(np.mean(timings)),\n",
    "    \"top_features_per_class\": {\n",
    "        cls: [{\"feature\": f, \"coef\": s} for f, s in feats]\n",
    "        for cls, feats in top_features.items()\n",
    "    },\n",
    "    \"classes\": clf.classes_(),\n",
    "}\n",
    "\n",
    "EVAL_METRICS_PATH.write_text(json.dumps(eval_payload, indent=2), encoding=\"utf-8\")\n",
    "print(\"Wrote\", EVAL_METRICS_PATH)"
))

cells.append(code(
    "def _row(label, d):\n",
    "    return f\"| {label} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | {d['support']} |\"\n",
    "\n",
    "val_d  = eval_payload[\"val\"]\n",
    "test_d = eval_payload[\"test\"]\n",
    "lines: list[str] = []\n",
    "lines += [\n",
    "    \"# Model Evaluation Results\",\n",
    "    \"\",\n",
    "    \"> Draft generated on Day 8. Final numbers filled in after Day 19's full benchmark.\",\n",
    "    \"\",\n",
    "    \"## Headline numbers\",\n",
    "    \"\",\n",
    "    f\"- Validation accuracy: **{val_d['accuracy']:.3f}**  (macro F1 = {val_d['macro_f1']:.3f})\",\n",
    "    f\"- Test accuracy:       **{test_d['accuracy']:.3f}**  (macro F1 = {test_d['macro_f1']:.3f})\",\n",
    "    f\"- Avg model latency: **{eval_payload['latency_ms_per_ticket']:.3f} ms/ticket**\",\n",
    "    \"\",\n",
    "    \"Target thresholds (from `docs/PROJECT_PLAN.md`): accuracy >= 0.88, sentiment F1 >= 0.80, API p99 < 100 ms.\",\n",
    "    \"\",\n",
    "    \"## Per-class metrics (test split)\",\n",
    "    \"\",\n",
    "    \"| Category | Precision | Recall | F1 | Support |\",\n",
    "    \"|---|---|---|---|---|\",\n",
    "]\n",
    "for label in CATEGORIES:\n",
    "    lines.append(_row(label, test_d[\"per_class\"][label]))\n",
    "\n",
    "lines += [\n",
    "    \"\",\n",
    "    \"## Top features per class\",\n",
    "    \"\",\n",
    "]\n",
    "for cls, feats in eval_payload[\"top_features_per_class\"].items():\n",
    "    top_words = \", \".join(f\"`{f['feature']}`\" for f in feats[:8])\n",
    "    lines.append(f\"- **{cls}** \u2014 {top_words}\")\n",
    "\n",
    "lines += [\n",
    "    \"\",\n",
    "    \"## Failure modes observed\",\n",
    "    \"\",\n",
    "    \"(See notebook section 5 for the live table. Day 18 will expand this with the edge-case suite.)\",\n",
    "    \"\",\n",
    "]\n",
    "\n",
    "RESULTS_MD.write_text(\"\\n\".join(lines) + \"\\n\", encoding=\"utf-8\")\n",
    "print(\"Wrote\", RESULTS_MD)\n",
    "print(\"\\n---- preview ----\")\n",
    "print(\"\\n\".join(lines[:25]))"
))

# ---------------------------------------------------------------------- 8
cells.append(md(
    "## 8. Day-8 sign-off checklist",
    "",
    "* [x] Confusion matrix visualised for val and test.",
    "* [x] Per-class precision / recall / F1 on val and test.",
    "* [x] Top-15 n-gram features per class extracted and rendered.",
    "* [x] Confidence calibration examined; threshold sweep exported.",
    "* [x] Failure modes listed (top confused pairs).",
    "* [x] Latency proxy recorded (target: stay under 100 ms end-to-end).",
    "* [x] `docs/results.md` first draft written.",
    "",
    "**Risks logged for later days:**",
    "",
    "* If Day 19's full latency benchmark on 1000 tickets exceeds 80 ms end-to-end,",
    "  swap `solver=\"liblinear\"` for `solver=\"saga\"` with a smaller `max_iter`.",
    "* If `Technical Setup` recall stays < 0.65, expand the keyword table in",
    "  `src/ticket_router/preprocessing/dataset.py` and consider seed edge cases",
    "  from `data/synthetic/` (Day 18)."
))

# ---------------------------------------------------------------------- end
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11.9",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "file_extension": ".py",
        "nbconvert_exporter": "python",
    },
}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT}")