"""Prepare the dataset: load raw CSV, map to 5 categories, stratified split.

Reads:   data/raw/tickets_raw.csv
Writes:  data/processed/{train,val,test}.csv
Prints:  per-split category distributions and total counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ticket_router.preprocessing.dataset import (
    build_labeled_frame,
    category_distribution,
    save_splits,
    stratified_split,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "raw" / "tickets_raw.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed"


def prepare(raw_path: Path, out_dir: Path, seed: int = 42) -> dict[str, object]:
    labeled = build_labeled_frame(raw_path, english_only=True)
    train, val, test = stratified_split(labeled, seed=seed)
    train_p, val_p, test_p = save_splits(train, val, test, out_dir)

    summary = {
        "raw_path": str(raw_path),
        "out_dir": str(out_dir),
        "total_labeled": int(len(labeled)),
        "train_path": str(train_p),
        "val_path": str(val_p),
        "test_path": str(test_p),
        "train_size": int(len(train)),
        "val_size": int(len(val)),
        "test_size": int(len(test)),
        "overall_distribution": category_distribution(labeled),
        "train_distribution": category_distribution(train),
        "val_distribution": category_distribution(val),
        "test_distribution": category_distribution(test),
        "seed": seed,
    }
    return summary


def _print_summary(summary: dict[str, object]) -> None:
    print(f"Loaded {summary['total_labeled']} labeled rows from {summary['raw_path']}")
    print(f"Wrote splits to {summary['out_dir']}")
    print(
        f"  train: {summary['train_size']}\n"
        f"  val:   {summary['val_size']}\n"
        f"  test:  {summary['test_size']}"
    )
    print("Overall category distribution:")
    for cat, count in summary["overall_distribution"].items():
        print(f"  {cat:<18s} {count}")
    print("Train distribution:")
    for cat, count in summary["train_distribution"].items():
        print(f"  {cat:<18s} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="Path to raw CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    summary = prepare(args.raw, args.out, seed=args.seed)
    _print_summary(summary)


if __name__ == "__main__":
    main()