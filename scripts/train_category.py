"""Train the TF-IDF + LogReg category classifier on the prepared splits.

Usage (from repo root):

    python scripts/train_category.py

Outputs (gitignored, regenerated each run):

    artifacts/category_model.joblib   - full sklearn Pipeline
    artifacts/train_metrics.json      - per-class precision/recall/F1 + accuracy

The script reads `data/processed/{train,val}.csv` produced by
`scripts/prepare_data.py`. It reports both train and val metrics so we can
spot over-fitting early.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ticket_router.models.category_classifier import (  # noqa: E402
    CATEGORIES,
    CategoryClassifier,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "category_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "train_metrics.json"


def _load_split(name: str) -> tuple[list[str], list[str]]:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/prepare_data.py` first."
        )
    df = pd.read_csv(path)
    if "text" not in df.columns or "category" not in df.columns:
        raise ValueError(f"{path} must have 'text' and 'category' columns")
    return df["text"].astype(str).tolist(), df["category"].astype(str).tolist()


def _metrics_report(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    labels = list(CATEGORIES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "classification_report": report,
    }


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train/val splits...")
    train_texts, train_labels = _load_split("train")
    val_texts, val_labels = _load_split("val")
    print(f"  train: {len(train_texts):,} rows")
    print(f"  val:   {len(val_texts):,} rows")

    print("Training TF-IDF (word 1-2 + char 3-5) + LogReg(balanced)...")
    t0 = time.perf_counter()
    clf = CategoryClassifier.build().fit(train_texts, train_labels)
    train_seconds = time.perf_counter() - t0
    print(f"  fit time: {train_seconds:.2f}s")

    print("Evaluating on train + val...")
    train_pred = clf.predict(train_texts)
    val_pred = clf.predict(val_texts)
    metrics = {
        "train": _metrics_report(train_labels, train_pred),
        "val": _metrics_report(val_labels, val_pred),
        "fit_seconds": train_seconds,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "classes": clf.classes_(),
    }

    val_acc = metrics["val"]["accuracy"]
    val_macro = metrics["val"]["macro_f1"]
    print(f"  val accuracy : {val_acc:.4f}")
    print(f"  val macro F1 : {val_macro:.4f}")
    print("  val per-class:")
    for label, row in metrics["val"]["per_class"].items():
        print(
            f"    {label:<18} P={row['precision']:.3f}  "
            f"R={row['recall']:.3f}  F1={row['f1']:.3f}  "
            f"support={row['support']}"
        )

    print(f"Saving pipeline to {MODEL_PATH}...")
    clf.save(MODEL_PATH)

    print(f"Saving metrics to {METRICS_PATH}...")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
