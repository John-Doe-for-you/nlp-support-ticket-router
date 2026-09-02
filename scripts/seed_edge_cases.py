import json
import sys
from ticket_router.db.database import get_session_factory, init_db
from ticket_router.db.repository import save_classification
from ticket_router.pipeline.inference import PredictionResult

def seed_edge_cases(file_path: str):
    """Load tickets from a JSONL file and save them to the database.
    
    Since we don't want to run the full ML pipeline (and we want to 
    test specifically how the DB handles these seeds), we mock the 
    PredictionResult with the expected values from the JSONL.
    """
    print(f"Seeding edge cases from {file_path}...")
    
    init_db()
    factory = get_session_factory()
    
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            # Create a mock PredictionResult based on expected values
            # In a real scenario, we'd run the pipeline. Here, we seed
            # the 'ground truth' as the prediction for testing purposes.
            result = PredictionResult(
                ticket_id=f"edge_{count:03d}",
                text=data["text"],
                customer_plan=data["customer_plan"],
                customer_id=data["customer_id"],
                category=data["expected_category"],
                category_confidence=0.95,
                sentiment="Neutral", # Simplified for seeding
                sentiment_scores={"neg": 0.1, "neu": 0.8, "pos": 0.1, "compound": 0.0},
                priority=data["expected_priority"],
                priority_score=80 if data["expected_priority"] == "P1" else 50,
                priority_breakdown={"total": 80 if data["expected_priority"] == "P1" else 50},
                routed_to="test-team",
                urgency_signals=[],
                latency_ms=10
            )
            
            with factory() as session:
                save_classification(session, result, commit=True)
                count += 1
    
    print(f"Successfully seeded {count} edge cases.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/seed_edge_cases.py <path_to_jsonl>")
        sys.exit(1)
    
    seed_edge_cases(sys.argv[1])
