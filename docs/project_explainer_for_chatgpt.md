# Customer Intelligence Platform: Complete Explanation and Interview Prep

Use this document to understand the project end to end or paste it into ChatGPT for concept revision.

## 1. One-Minute Summary

This project is a Customer Intelligence Platform with two AI services. The first service is an ML model that predicts whether a bank customer will subscribe to a term deposit. The second service is a RAG-style complaint intelligence system that retrieves evidence from CFPB complaint narratives and summarizes complaint themes with cited record IDs. The integrated endpoint combines both: it returns a customer's conversion band and the top complaint themes relevant to optional filters.

The project demonstrates MLOps and LLMOps practices: data ingestion, validation, feature engineering, model tracking, promotion gates, API serving, retrieval indexing, RAG evaluation, monitoring, Docker, and CI/CD.

## 2. What Problem It Solves

A marketing or customer intelligence team needs to answer two questions:

1. Is this customer likely to convert if contacted for a term-deposit campaign?
2. What complaint themes are common around products/issues relevant to this customer segment?

The project answers both with one integrated API:

```text
POST /customer-intel
```

It returns:

- conversion probability
- conversion band: low, medium, high
- top complaint themes
- cited complaint record IDs
- model version
- index version

## 3. Datasets

### UCI Bank Marketing Dataset

Purpose: train the campaign conversion prediction model.

Target:

```text
y = whether the customer subscribed to a term deposit
```

Important fields:

- age
- job
- marital
- education
- balance
- housing
- loan
- contact
- duration
- campaign
- pdays
- previous
- poutcome

### CFPB Consumer Complaint Database

Purpose: build a complaint retrieval index.

Used fields:

- complaint ID
- date received
- product
- company
- issue
- complaint narrative

PII safety decision:

High-risk columns like ZIP code, state, tags, submitted channel details, and broad free-form responses are not used for indexing. The narrative text is also sanitized for obvious emails, phone-like numbers, ZIP codes, and account-like numbers.

## 4. Data Pipeline

### `src/data_pipeline/ingest.py`

Responsibilities:

- Downloads UCI data from its public URL.
- Downloads CFPB complaint data from the public CFPB file.
- Falls back to synthetic CFPB data only if the public download is unavailable.
- Saves local data under `data/`.
- Writes SHA-256 hash sidecar files for reproducibility.

Why hashes matter:

Hashes document the exact data version used for training/evaluation without committing raw data into Git.

### `src/data_pipeline/validate.py`

Uses Pandera to enforce:

- exact column names
- strict data types
- non-null rules
- business rules

Business rules:

- age must be between 18 and 95
- balance cannot be null
- duration must be greater than 0
- pdays must be -1 or positive
- campaign must be positive

Why this matters:

Validation catches bad data before training, which prevents silent model-quality issues.

### `src/data_pipeline/features.py`

Feature functions:

- `encode_categoricals`
- `bin_age`
- `compute_contact_features`
- `scale_numerics`
- `get_feature_names`
- `build_features`

Key design:

The scaler and categorical mappings are reused during serving. This prevents training-serving skew.

## 5. ML Training

### Baseline Model

Model:

```text
LogisticRegression(C=1.0, max_iter=1000)
```

Why:

It is simple, interpretable, and useful as a baseline.

### Improved Model

Model:

```text
XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05)
```

Why:

XGBoost handles nonlinear relationships and mixed feature interactions better than logistic regression.

### MLflow Tracking

Logged items:

- model parameters
- dataset hash
- ROC-AUC
- PR-AUC
- F1
- Brier score
- confusion matrix values
- threshold analysis at 0.3, 0.4, 0.5
- inference latency
- model artifact
- scaler artifact
- categorical mappings
- feature names
- feature importance plot
- calibration curve

Why MLflow matters:

MLflow makes experiments reproducible and lets the serving code load the same model and preprocessing artifacts used during training.

## 6. Model Evaluation and Promotion Gate

Metrics:

- ROC-AUC: ranking quality across thresholds
- PR-AUC: better for imbalanced positive class problems
- F1: balance between precision and recall at a chosen threshold
- Confusion matrix: counts true/false positives and negatives
- Brier score: calibration quality
- Latency: serving feasibility

Promotion rule:

```text
PROMOTE if:
improved PR-AUC >= baseline PR-AUC + 0.03
improved F1 >= baseline F1 - 0.02
single-row inference latency <= 200ms
```

Why this is realistic:

Production model promotion should not rely on "new model feels better." It should pass a measurable gate against the current baseline.

## 7. FastAPI ML Service

Main file:

```text
src/serving/serve.py
```

Endpoints:

- `GET /health`
- `POST /predict`
- `POST /batch-score`
- `GET /metrics`
- `POST /customer-intel`

Pydantic schemas:

```text
src/serving/schemas.py
```

What schemas do:

- validate input types
- enforce business constraints
- reject malformed requests with clear 422 errors

Model loading:

```text
src/serving/model_loader.py
```

It loads:

- MLflow model
- scaler
- categorical mappings
- run ID
- timestamp

## 8. RAG System

### `src/rag/build_index.py`

Steps:

1. Load CFPB sample.
2. Strip risky PII columns.
3. Sanitize narratives.
4. Split text into sentence-aware chunks.
5. Embed chunks.
6. Build FAISS inner-product index.
7. Save `data/faiss.index`.
8. Save `data/chunk_map.json`.

Embedding note:

The intended embedding model is `sentence-transformers/all-MiniLM-L6-v2`. The project includes a local hashing fallback because the Windows Python 3.12 environment had a Torch DLL issue. This makes the project reproducible even without a working Torch install.

### `src/rag/retrieve.py`

Responsibilities:

- load FAISS index
- embed query
- retrieve top chunks
- apply product/company/issue/date filters
- refuse if best score is below threshold
- log every retrieval call to JSONL

### `src/rag/answer.py`

Endpoint:

```text
POST /ask-complaints
```

Behavior:

- retrieves evidence
- refuses if retrieval is weak
- answers only from evidence
- returns evidence IDs
- reports retrieval and generation latency

Guardrails:

- token budget guardrail trims chunks if prompt exceeds 1,500 tokens
- latency guardrail logs a warning if generation exceeds 3 seconds
- prompt instructs the assistant not to give legal or financial advice

## 9. RAG Evaluation

File:

```text
src/rag/rag_eval.py
```

Stretch version includes 20 cases:

- 10 standard questions
- 5 adversarial questions
- 5 edge cases

Report:

```text
docs/rag_eval_report.md
```

Metrics:

- category
- pass/fail
- retrieval score
- latency
- token count
- refused flag
- reason

Why this matters:

RAG quality must be evaluated for both retrieval usefulness and refusal behavior. A system that always answers can be dangerous; a system that refuses appropriately is safer.

## 10. Monitoring

### ML Drift

File:

```text
monitoring/ml_drift.py
```

Simulated drift:

- add 15 years to 30% of ages
- flip 20% of outcomes
- introduce 5% missing balance values

Output:

```text
monitoring/ml_drift_report.html
```

### RAG Monitoring

File:

```text
monitoring/rag_monitor.py
```

Computes:

- retrieval hit rate
- empty retrieval count
- average top-1 similarity
- refusal rate
- average token count
- average generation latency
- average total latency

Output:

```text
monitoring/rag_monitoring_report.json
```

## 11. CI/CD

### CI Workflow

File:

```text
.github/workflows/ci.yml
```

Runs:

- checkout
- set up Python 3.10
- install dependencies
- ingest sample data
- validate data
- run pytest

### Evaluation Gate Workflow

File:

```text
.github/workflows/eval_gate.yml
```

Runs:

- model evaluation gate
- fails if the model is blocked

Why this matters:

CI/CD prevents broken validation, tests, or promotion logic from silently entering the main branch.

## 12. Docker and Deployment

### Dockerfile

Production-style features:

- multi-stage build
- non-root user
- healthcheck
- exposed port 8000
- Uvicorn command

### Docker Compose

Runs:

- `ml-api` on port 8000
- `rag-api` on port 8001

Both services share the local `data/` volume for FAISS and local artifacts.

### Cloud Run

Recommended settings:

- min instances: 0
- max instances: 1
- memory: 512Mi
- region: us-central1

Why:

These settings reduce cost risk and fit the free-tier style requirement.

## 13. How To Explain This In An Interview

### Short Answer

"I built an end-to-end customer intelligence platform with an ML service and a RAG service. The ML service predicts term-deposit conversion using UCI Bank Marketing data and tracks experiments in MLflow. The RAG service indexes CFPB complaint narratives in FAISS and answers complaint questions using retrieved evidence. I added validation, promotion gates, FastAPI endpoints, Docker Compose, CI/CD, drift monitoring, RAG monitoring, and documentation."

### Strong Technical Answer

"The important engineering decision was to avoid a notebook-only workflow. I created reusable feature functions, saved the scaler and categorical mappings as MLflow artifacts, and loaded those same artifacts in serving to prevent training-serving skew. For RAG, I stripped risky PII fields before indexing, persisted a FAISS index and chunk map, implemented similarity-threshold refusal, and created a 20-case evaluation set including adversarial and edge-case prompts. The system is testable through pytest and deployable through Docker Compose or Cloud Run."

## 14. Questions You May Be Asked

### Why did you use PR-AUC?

The positive class is relatively rare, so PR-AUC is more informative than accuracy. It focuses on how well the model identifies likely subscribers without being dominated by the majority "no" class.

### Why save the scaler and mappings?

If serving recomputes preprocessing differently from training, predictions become unreliable. Saving preprocessing artifacts with MLflow ensures the API uses the exact same transformations as training.

### Why use a promotion gate?

A promotion gate converts model release into a measurable decision. It prevents deploying a model unless it improves PR-AUC enough, preserves F1, and meets latency requirements.

### Why FAISS?

FAISS provides efficient vector similarity search. It is suitable for retrieving the most relevant complaint chunks from thousands of embedded narratives.

### What is the refusal logic?

If the best retrieval score is below the threshold, the system refuses to answer and asks the user to refine the question or filters. This prevents unsupported answers.

### What would you improve next?

I would fix the local Python/Torch setup to use MiniLM embeddings directly, add stronger PII redaction, use a real hosted metrics store, add API authentication, and deploy with a proper model registry alias.

## 15. Exact Demo Commands

```powershell
python src\data_pipeline\validate.py
python src\training\evaluate.py
python src\rag\build_index.py
python src\rag\rag_eval.py
python -m pytest tests\ -q
python monitoring\ml_drift.py
python monitoring\rag_monitor.py
```

Start services:

```powershell
python -m uvicorn src.serving.serve:app --host 127.0.0.1 --port 8000
python -m uvicorn src.rag.answer:app --host 127.0.0.1 --port 8001
```

Open API docs:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8001/docs
```

This is the closest thing to a UI for the project: FastAPI automatically provides interactive Swagger documentation for every endpoint.
