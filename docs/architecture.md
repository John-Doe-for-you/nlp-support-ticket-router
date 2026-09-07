# Architecture

This diagram illustrates the end-to-end flow of the NLP Support Ticket Router,
from API request to ticket routing.

```mermaid
flowchart TB
    %% Client & API layer
    subgraph API["FastAPI Application"]
        direction LR
        A[POST /classify] -- text, customer_plan, customer_id --> B[Request Timing Middleware]
        B --> C[Pydantic Validation]
        C --> D[InferencePipeline]
        D --> E[/health Probe]
    end

    %% ML Pipeline
    subgraph Pipeline["Inference Pipeline"]
        direction TB
        F[Text Cleaner] -- lowercasing, strip HTML/URLs, normalize whitespace --> G[TF-IDF Vectorizer]
            G uses: word n-grams (1-2), char n-grams (3-5)
        G --> H[Category Classifier (LogReg)]
            H returns: category, category_confidence
        H --> I[Category Result]
        F -->|shared cleaned text--> J[VADER Sentiment Analyzer]
        J -->|sentiment + scores--> K[Custom Urgency Lexicon]
        K -->|negative intensity delta--> L[Priority Engine]
    end

    %% Priority & Routing
    subgraph Routing["Priority & Team Routing"]
        direction TB
        L --> M[Priority Score]
            M -->|score >= 70| N[P1 - Critical]
            M -->|40 <= score < 70| O[P2 - Standard]
            M -->|score < 40| P[P3 - Low]
        M --> O[Team Router]
        O -->|Category -> Team| Q[routed_to: billing-team / identity-team / engineering-team / product-team / support-team]
    end

    %% DB Layer
    subgraph DB["SQLite + SQLAlchemy"]
        direction TB
        R[Predictions Table] -- persist every classification --> S[Tickets Table]
        R -->|ticket_id, category, sentiment, priority, routed_to| T[Repository Pattern]
    end

    %% Connections
    C -->|persist=False/True| R
    E -->|model_loaded, db_ready| U[/health]
    Q --> V[Dashboard / Client]

    style API fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Pipeline fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Routing fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style DB fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```