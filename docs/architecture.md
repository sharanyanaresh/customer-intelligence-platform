# Architecture

```mermaid
flowchart LR
    A["Public data sources<br/>UCI Bank Marketing + CFPB complaints"] --> B["src/data_pipeline/ingest.py<br/>download/sample + SHA-256 hashes"]
    B --> C["src/data_pipeline/validate.py<br/>Pandera schema + business rules"]
    C --> D["src/data_pipeline/features.py<br/>feature engineering + reusable scaler"]
    D --> E["src/training/train.py<br/>baseline + improved models"]
    E --> F["MLflow tracking<br/>models, scaler, mappings, plots, metrics"]
    E --> G["src/training/evaluate.py<br/>promotion gate"]
    G -->|PROMOTED| F
    G -->|BLOCKED| H["Stop merge / retrain"]
    F --> I["src/serving/model_loader.py<br/>load versioned model bundle"]
    I --> J["src/serving/serve.py<br/>FastAPI ML API"]
    J --> K["/predict<br/>/batch-score<br/>/metrics<br/>/health"]
    B --> L["src/rag/build_index.py<br/>PII stripping + chunking"]
    L --> M["FAISS index<br/>data/faiss.index + chunk_map.json"]
    M --> N["src/rag/retrieve.py<br/>similarity search + refusal logic"]
    N --> O["src/rag/answer.py<br/>/ask-complaints"]
    J --> P["/customer-intel<br/>ML score + complaint themes + citations"]
    N --> P
    Q["GitHub Actions CI<br/>ingest + validate + pytest"] --> C
    R["GitHub Actions eval gate<br/>promotion check"] --> G
    S["Dockerfile + docker-compose.yml<br/>local production spine"] --> J
    S --> O
    T["monitoring/ml_drift.py<br/>Evidently drift report"] --> U["Retrain trigger"]
    V["monitoring/rag_monitor.py<br/>hit rate + refusal + latency"] --> W["LLMOps monitoring report"]
```

The system is built as two services sharing one production spine: the ML API scores campaign conversion, while the RAG API retrieves complaint evidence. MLflow, FAISS, CI, monitoring, and Docker Compose make the project reproducible from ingestion through demo.

