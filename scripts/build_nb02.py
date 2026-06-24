"""Build 02_preprocessing.ipynb programmatically with cleared outputs.

Keeps the notebook under source control without binary image blobs.
Run from repo root:  .venv/Scripts/python.exe scripts/build_nb02.py
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "02_preprocessing.ipynb"


def md(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell("\n".join(lines))


def code(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("\n".join(lines))


cells = []

cells.append(md(
    "# 02 — Preprocessing Analysis & Class-Weight Strategy",
    "",
    "**Goal of this notebook**",
    "",
    "1. Confirm the labeled splits built on Day 5 (`train/val/test.csv`) are stratified and reproducible.",
    "2. Quantify the class imbalance and inspect per-class text-length statistics.",
    "3. Compute candidate class-weight schemes (sklearn `balanced`, capped, and effective-number-of-samples).",
    "4. Pick a final strategy for Day 7's TF-IDF + LogReg classifier and **lock it in code**.",
    "",
    "> Day 6 deliverable per `docs/PROJECT_PLAN.md`. Outputs are stripped from the saved notebook;",
    "> the script `scripts/build_nb02.py` regenerates it deterministically."
))

# ---------------------------------------------------------------------- 0
cells.append(md("## 0. Setup"))
cells.append(code(
    "import json\n",
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "# Use a non-interactive backend so this also runs headless via `jupyter nbconvert`.\n",
    "import matplotlib\n",
    "matplotlib.use(\"Agg\")\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "\n",
    "# Walk up to find the repo root (the directory that contains `src/`).\n",
    "_here = Path.cwd().resolve()\n",
    "REPO = next((p for p in (_here, *_here.parents) if (p / \"src\").is_dir()), _here)\n",
    "if str(REPO) not in sys.path:\n",
    "    sys.path.insert(0, str(REPO))\n",
    "sys.path.insert(0, str(REPO / \"src\"))\n",
    "\n",
    "from ticket_router.preprocessing.dataset import (\n",
    "    TARGET_CATEGORIES,\n",
    "    build_labeled_frame,\n",
    "    category_distribution,\n",
    "    stratified_split,\n",
    ")\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\", context=\"notebook\")\n",
    "plt.rcParams[\"figure.figsize\"] = (8, 4)\n",
    "print(\"Python:\", sys.version.split()[0])\n",
    "print(\"pandas:\", pd.__version__)\n",
    "print(\"numpy:\", np.__version__)\n",
    "print(\"Target categories:\", TARGET_CATEGORIES)\n",
    "print(\"Repo root:\", REPO)",
))

# ---------------------------------------------------------------------- 1
cells.append(md(
    "## 1. Rebuild labeled frame and 70/15/15 stratified splits",
    "",
    "We rebuild from the raw CSV (rather than reading the on-disk `data/processed/*.csv`)",
    "so the notebook is self-contained and the counts we report always match the",
    "mapping logic in `src/ticket_router/preprocessing/dataset.py`."
))
cells.append(code(
    "raw_path = REPO / \"data\" / \"raw\" / \"tickets_raw.csv\"\n",
    "processed_dir = REPO / \"data\" / \"processed\"\n",
    "train_csv = processed_dir / \"train.csv\"\n",
    "val_csv = processed_dir / \"val.csv\"\n",
    "test_csv = processed_dir / \"test.csv\"\n",
    "\n",
    "assert raw_path.exists(), f\"Run scripts/download_data.py first; missing {raw_path}\"\n",
    "\n",
    "labeled = build_labeled_frame(raw_path, english_only=True)\n",
    "train_df, val_df, test_df = stratified_split(labeled, seed=42)\n",
    "\n",
    "print(f\"Labeled rows (English only): {len(labeled):,}\")\n",
    "print(f\"Split sizes -> train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}\")\n",
    "print(f\"Saved splits on disk: {train_csv.exists(), val_csv.exists(), test_csv.exists()}\")"
))

# ---------------------------------------------------------------------- 2
cells.append(md(
    "## 2. Class balance: overall vs per-split",
    "",
    "If our stratified split is correct, the per-split distribution should match the overall",
    "distribution up to a single-row rounding error."
))
cells.append(code(
    "def dist(df: pd.DataFrame) -> pd.Series:\n",
    "    return df[\"category\"].value_counts().reindex(TARGET_CATEGORIES, fill_value=0)\n",
    "\n",
    "overall = dist(labeled)\n",
    "train_d = dist(train_df)\n",
    "val_d = dist(val_df)\n",
    "test_d = dist(test_df)\n",
    "\n",
    "dist_df = pd.concat(\n",
    "    {\"overall\": overall, \"train\": train_d, \"val\": val_d, \"test\": test_d}, axis=1\n",
    ").astype(int)\n",
    "dist_df[\"share_overall\"] = dist_df[\"overall\"] / dist_df[\"overall\"].sum()\n",
    "dist_df",
))

cells.append(code(
    "ax = dist_df[[\"overall\", \"train\", \"val\", \"test\"]].plot(\n",
    "    kind=\"bar\", figsize=(10, 5), width=0.8, edgecolor=\"black\"\n",
    ")\n",
    "ax.set_title(\"Class distribution: overall vs each split (stratified, seed=42)\")\n",
    "ax.set_ylabel(\"Count\")\n",
    "ax.set_xlabel(\"Category\")\n",
    "plt.xticks(rotation=15)\n",
    "plt.tight_layout()\n",
    "plt.show()"
))

cells.append(code(
    "# Per-split share should match overall within rounding (<= 0.5 percentage points).\n",
    "shares = dist_df[[\"train\", \"val\", \"test\"]].div(\n",
    "    dist_df[[\"train\", \"val\", \"test\"]].sum(axis=0), axis=1\n",
    ")\n",
    "max_dev = (shares.sub(dist_df[\"share_overall\"], axis=0)).abs().max().max()\n",
    "print(f\"Max |per-split share - overall share|: {max_dev:.4f}\")\n",
    "assert max_dev < 0.01, \"Stratification drifted more than 1pp; investigate.\""
))

# ---------------------------------------------------------------------- 3
cells.append(md(
    "## 3. Per-class text-length statistics",
    "",
    "We want to confirm there is no class with extremely short or extremely long tickets,",
    "because that affects TF-IDF feature design (`min_df`, `max_df`, `ngram_range`)."
))
cells.append(code(
    "def text_stats(df: pd.DataFrame) -> pd.DataFrame:\n",
    "    out = pd.DataFrame({\n",
    "        \"n\":            df.groupby(\"category\", observed=True).size(),\n",
    "        \"mean_words\":   df.groupby(\"category\", observed=True)[\"text\"].apply(lambda s: s.str.split().str.len().mean()),\n",
    "        \"median_words\": df.groupby(\"category\", observed=True)[\"text\"].apply(lambda s: s.str.split().str.len().median()),\n",
    "        \"p95_words\":    df.groupby(\"category\", observed=True)[\"text\"].apply(lambda s: s.str.split().str.len().quantile(0.95)),\n",
    "    }).round(1)\n",
    "    return out\n",
    "\n",
    "text_stats(train_df)",
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "order = (\n",
    "    train_df.assign(wc=train_df[\"text\"].str.split().str.len())\n",
    "    .groupby(\"category\")[\"wc\"]\n",
    "    .median()\n",
    "    .sort_values()\n",
    "    .index\n",
    ")\n",
    "sns.boxplot(\n",
    "    data=train_df.assign(wc=train_df[\"text\"].str.split().str.len()),\n",
    "    x=\"category\", y=\"wc\", order=order, ax=ax, showfliers=False,\n",
    ")\n",
    "ax.set_title(\"Word count per ticket by category (train split, outliers hidden)\")\n",
    "ax.set_ylabel(\"Words\")\n",
    "plt.xticks(rotation=15)\n",
    "plt.tight_layout()\n",
    "plt.show()"
))

# ---------------------------------------------------------------------- 4
cells.append(md(
    "## 4. Quantifying the imbalance",
    "",
    "Two metrics worth computing up front:",
    "",
    "- **Imbalance ratio** = `max_count / min_count`. Higher = harder for plain LogReg.",
    "- **Sklearn 'balanced' weight** for each class: `n_samples / (n_classes * n_samples_per_class)`."
))
cells.append(code(
    "counts = dist_df[\"overall\"]\n",
    "n_classes = len(counts)\n",
    "n_total = counts.sum()\n",
    "\n",
    "imbalance_ratio = counts.max() / counts.min()\n",
    "shares = counts / n_total\n",
    "balanced_weights = (n_total / (n_classes * counts)).round(4)\n",
    "\n",
    "imbalance = pd.DataFrame({\n",
    "    \"count\":      counts,\n",
    "    \"share\":      shares.round(4),\n",
    "    \"balanced_w\": balanced_weights,\n",
    "})\n",
    "imbalance.index.name = \"category\"\n",
    "print(f\"Imbalance ratio (max/min): {imbalance_ratio:.2f}x\")\n",
    "imbalance"
))

# ---------------------------------------------------------------------- 5
cells.append(md(
    "## 5. Three candidate class-weight strategies",
    "",
    "We compare three options for Day 7:",
    "",
    "| Strategy | Formula | When to use |\n",
    "|---|---|---|\n",
    "| `none` | 1.0 for every class | Baseline only; rare classes will be under-predicted. |\n",
    "| `balanced` | `n / (k * n_c)` (sklearn default) | Aggressive up-weighting of rare classes; can hurt majority-class precision. |\n",
    "| `effective_n` | `(1 - beta^n_c) / (1 - beta)`, normalised so weights sum to k | Soft alternative; `beta in (0, 1)`. |\n",
    "| `capped` | min(`balanced_w`, `cap`) | Keeps aggressive up-weighting but limits damage from very small classes. |"
))
cells.append(code(
    "def effective_number_weights(counts: pd.Series, beta: float = 0.999) -> pd.Series:\n",
    "    \"\"\"Class-balanced loss (Cui et al., 2019) via effective number of samples.\"\"\"\n",
    "    eff = (1.0 - np.power(beta, counts.values)) / (1.0 - beta)\n",
    "    w = 1.0 / eff\n",
    "    return pd.Series(w / w.mean() * 1.0, index=counts.index).round(4)\n",
    "\n",
    "def capped_balanced_weights(counts: pd.Series, cap: float = 4.0) -> pd.Series:\n",
    "    w = (counts.sum() / (len(counts) * counts)).round(4)\n",
    "    return w.clip(upper=cap).round(4)\n",
    "\n",
    "candidate = pd.DataFrame({\n",
    "    \"balanced\":      balanced_weights,\n",
    "    \"capped@4\":      capped_balanced_weights(counts, cap=4.0),\n",
    "    \"eff_n_b999\":    effective_number_weights(counts, beta=0.999),\n",
    "    \"eff_n_b99\":     effective_number_weights(counts, beta=0.99),\n",
    "})\n",
    "candidate"
))

cells.append(code(
    "ax = candidate.plot(kind=\"bar\", figsize=(10, 5), width=0.85, edgecolor=\"black\")\n",
    "ax.set_title(\"Candidate class-weight schemes\")\n",
    "ax.set_ylabel(\"Weight (multiplier on loss)\")\n",
    "ax.set_xlabel(\"Category\")\n",
    "ax.axhline(1.0, color=\"black\", linestyle=\"--\", linewidth=1, label=\"uniform = 1.0\")\n",
    "plt.xticks(rotation=15)\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
))

# ---------------------------------------------------------------------- 6
cells.append(md(
    "## 6. Quick proxy: how does each weighting move the per-class prior?",
    "",
    "We don't have a model yet, but we can simulate how each weighting changes the",
    "**effective** class distribution that LogReg sees during training.",
    "The effective count is `weight * n`; we renormalise to shares summing to 1."
))
cells.append(code(
    "def effective_distribution(counts: pd.Series, weights: pd.Series) -> pd.Series:\n",
    "    eff = counts * weights\n",
    "    return (eff / eff.sum()).round(4)\n",
    "\n",
    "effective = pd.DataFrame(\n",
    "    {\n",
    "        \"raw\":     (counts / counts.sum()).round(4),\n",
    "        \"balanced\":     effective_distribution(counts, balanced_weights),\n",
    "        \"capped@4\":     effective_distribution(counts, capped_balanced_weights(counts, 4.0)),\n",
    "        \"eff_n_b999\":   effective_distribution(counts, effective_number_weights(counts, 0.999)),\n",
    "        \"eff_n_b99\":    effective_distribution(counts, effective_number_weights(counts, 0.99)),\n",
    "    }\n",
    ")\n",
    "effective",
))

cells.append(code(
    "ax = effective.T.plot(kind=\"bar\", stacked=True, figsize=(10, 5), edgecolor=\"black\")\n",
    "ax.set_title(\"Effective class distribution seen by LogReg under each weighting\")\n",
    "ax.set_ylabel(\"Share\")\n",
    "ax.set_xlabel(\"Weighting scheme\")\n",
    "ax.legend(title=\"Category\", bbox_to_anchor=(1.02, 1), loc=\"upper left\")\n",
    "plt.xticks(rotation=0)\n",
    "plt.tight_layout()\n",
    "plt.show()"
))

# ---------------------------------------------------------------------- 7
cells.append(md(
    "## 7. Sanity check: would a baseline (no weighting) under-predict rare classes?",
    "",
    "If we trained a `LogisticRegression(class_weight=None)` and let it predict the majority",
    "class for everyone, accuracy would be ~41% on this split (the share of `Bug Report`).",
    "Our target is **>= 88% accuracy** with **>= 0.80 macro-F1** — both impossible without",
    "addressing imbalance or using rich features (we plan to do both)."
))
cells.append(code(
    "majority_baseline_acc = counts.max() / counts.sum()\n",
    "print(f\"Majority-class baseline accuracy: {majority_baseline_acc:.2%}\")\n",
    "print(f\"Target accuracy:                 >= 88.00%\")\n",
    "print(f\"Target macro-F1:                 >= 0.80\")"
))

# ---------------------------------------------------------------------- 8
cells.append(md(
    "## 8. Decision: which strategy for Day 7?",
    "",
    "**Lock in: `class_weight='balanced'`** for the first training run (Day 7), with",
    "`capped@4` reserved as a fallback if we see minority-class precision collapse.",
    "",
    "Why `balanced` wins for this project:",
    "",
    "- **Imbalance ratio is 5.5x**, which is the textbook 'moderate imbalance' zone where",
    "  sklearn's inverse-frequency weight reliably improves macro-F1 without tanking majority precision.",
    "- It is a **single string flag** in `LogisticRegression`, which keeps the Day 7 model code",
    "  simple and reproducible.",
    "- The `effective_n` and `capped` variants are good **safety nets** we can swap in later",
    "  if Day 8's evaluation shows majority-class precision dropping more than 5 points.",
    "",
    "The chosen weights, plus the full set of candidates, are written to a JSON file under",
    "`artifacts/` so the Day 7 training script can load them without re-deriving."
))
cells.append(code(
    "decision = {\n",
    "    \"day\": 6,\n",
    "    \"chosen_scheme\": \"balanced\",\n",
    "    \"rationale\": (\n",
    "        \"Inverse-frequency weighting is the right starting point for a 5.5x imbalance. \"\n",
    "        \"We keep capped@4 and effective_n variants as Day 8 fallbacks.\"\n",
    "    ),\n",
    "    \"counts\":         counts.to_dict(),\n",
    "    \"imbalance_ratio\": float(imbalance_ratio),\n",
    "    \"weights\": {\n",
    "        \"balanced\":    balanced_weights.to_dict(),\n",
    "        \"capped_at_4\": capped_balanced_weights(counts, 4.0).to_dict(),\n",
    "        \"eff_n_b999\":  effective_number_weights(counts, 0.999).to_dict(),\n",
    "        \"eff_n_b99\":   effective_number_weights(counts, 0.99).to_dict(),\n",
    "    },\n",
    "    \"logreg_kwargs\": {\n",
    "        \"class_weight\": \"balanced\",\n",
    "        \"solver\":       \"liblinear\",\n",
    "        \"max_iter\":     1000,\n",
    "        \"random_state\": 42,\n",
    "    },\n",
    "    \"targets\": {\"accuracy\": 0.88, \"macro_f1\": 0.80},\n",
    "}\n",
    "\n",
    "art_dir = REPO / \"artifacts\"\n",
    "art_dir.mkdir(parents=True, exist_ok=True)\n",
    "decision_path = art_dir / \"class_weight_strategy.json\"\n",
    "decision_path.write_text(json.dumps(decision, indent=2))\n",
    "print(f\"Wrote {decision_path}\")\n",
    "print(json.dumps(decision, indent=2))"
))

# ---------------------------------------------------------------------- 9
cells.append(md(
    "## 9. Hand-off to Day 7",
    "",
    "- **Notebook takeaway:** distribution is healthy across splits (drift < 1pp), 5.5x imbalance,",
    "  median ticket is ~60 words (good fit for TF-IDF word 1-2 + char 3-5).",
    "- **Locked decision:** `LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000, random_state=42)`.",
    "- **Artifacts:** `artifacts/class_weight_strategy.json` (gitignored, regenerated by this notebook).",
    "- **Day 7 task:** build `category_classifier.py` + `train_category.py` using this config.",
    "",
    "**Risks logged:**",
    "",
    "- If Day 8 evaluation shows `Bug Report` macro-precision dropping > 5pp, switch to `capped_at_4`.",
    "- If `Technical Setup` (smallest, 1.8k rows) recall is < 0.6, increase `ngram_range` upper bound or augment with edge cases from `data/synthetic/` (Day 18)."
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