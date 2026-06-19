"""Download raw dataset from HuggingFace. Populated on Day 3."""
from pathlib import Path
from datasets import load_dataset
import pandas as pd


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")

    # The dataset may have a train split or multiple splits; combine all
    if isinstance(dataset, dict):
        df = pd.concat(
            [dataset[split].to_pandas() for split in dataset],
            ignore_index=True,
        )
    else:
        df = dataset.to_pandas()

    out_path = raw_dir / "tickets_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()