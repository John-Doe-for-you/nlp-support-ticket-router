import time
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from ticket_router.pipeline.inference import get_default_pipeline

def evaluate_model(test_csv: str):
    """Run full inference on test set and report metrics and latency."""
    print(f"Loading test data from {test_csv}...")
    df = pd.read_csv(test_csv)
    
    if len(df) == 0:
        print("Error: Test CSV is empty.")
        return

    pipeline = get_default_pipeline()
    pipeline.load_category_model("artifacts/category_model.joblib")
    
    # We only need a subset for latency benchmarking if the file is huge, 

    # but for accuracy we use the full set.
    texts = df['text'].tolist()
    true_categories = df['category'].tolist()
    
    print(f"Evaluating {len(texts)} tickets...")
    
    preds = []
    latencies = []
    
    # To avoid loading everything into memory at once for very large sets, 
    # we iterate. We use a sample of 1000 for latency if the dataset is larger.
    latency_sample_size = 1000
    latency_indices = np.random.choice(len(texts), min(len(texts), latency_sample_size), replace=False)
    
    for i, text in enumerate(texts):
        t0 = time.perf_counter()
        result = pipeline.predict(text)
        t1 = time.perf_counter()
        
        preds.append(result.category)
        if i in latency_indices:
            latencies.append((t1 - t0) * 1000)

    # Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(true_categories, preds))
    
    acc = accuracy_score(true_categories, preds)
    print(f"Overall Accuracy: {acc:.2%}")
    
    # Latency Report
    if latencies:
        print("\n--- Latency Benchmark (Sample of 1000) ---")
        print(f"Mean Latency: {np.mean(latencies):.2f} ms")
        print(f"p50 Latency: {np.median(latencies):.2f} ms")
        print(f"p95 Latency: {np.percentile(latencies, 95):.2f} ms")
        print(f"p99 Latency: {np.percentile(latencies, 99):.2f} ms")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate.py <path_to_test_csv>")
        sys.exit(1)
    
    evaluate_model(sys.argv[1])
