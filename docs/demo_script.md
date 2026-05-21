# Demo Script: 5-8 Minutes

## Minute 1: Repository + Setup

Show the repo structure, `README.md`, and `.env.example`. Explain that raw data and generated artifacts are intentionally excluded from Git.

Commands:

```powershell
Get-ChildItem
Get-Content .\README.md
```

## Minute 2: Data Validation

Run ingestion only if data is missing, then validate the UCI dataset.

```powershell
python .\src\data_pipeline\validate.py
```

Expected highlight:

```text
Validation passed - 45208 rows, 17 columns.
```

## Minute 3: MLflow + Gate

Show the training output and promotion gate.

```powershell
python .\src\training\evaluate.py
mlflow ui
```

Highlight the baseline and improved runs, PR-AUC delta, feature importance, calibration curve, and blocked degraded model.

## Minute 4: ML API

Start the ML service:

```powershell
python -m uvicorn src.serving.serve:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"age\":42,\"job\":\"management\",\"marital\":\"married\",\"education\":\"tertiary\",\"default\":\"no\",\"balance\":1200,\"housing\":\"yes\",\"loan\":\"no\",\"contact\":\"cellular\",\"day\":15,\"month\":\"may\",\"duration\":180,\"campaign\":2,\"pdays\":-1,\"previous\":0,\"poutcome\":\"unknown\"}"
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"age\":42,\"job\":\"management\"}"
```

Show the valid score and the invalid `422` response.

## Minute 5: RAG API

Start the RAG service:

```powershell
python -m uvicorn src.rag.answer:app --host 127.0.0.1 --port 8001
```

Ask a complaint question:

```powershell
$body = @{ question = "What themes appear around debt not owed?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8001/ask-complaints" -Method Post -ContentType "application/json" -Body $body
```

Highlight the answer, evidence IDs, refusal logic, and `rag_eval_report.md`.

## Minute 6: Integrated Endpoint

Show the integrated endpoint:

```powershell
curl.exe -X POST http://127.0.0.1:8000/customer-intel -H "Content-Type: application/json" -d "{\"customer\":{\"age\":42,\"job\":\"management\",\"marital\":\"married\",\"education\":\"tertiary\",\"default\":\"no\",\"balance\":1200,\"housing\":\"yes\",\"loan\":\"no\",\"contact\":\"cellular\",\"day\":15,\"month\":\"may\",\"duration\":180,\"campaign\":2,\"pdays\":-1,\"previous\":0,\"poutcome\":\"unknown\"},\"product\":\"Credit reporting\",\"issue\":\"Incorrect information\"}"
```

Explain that it returns conversion band, probability, complaint themes, and cited complaint IDs.

## Minute 7: CI/CD

Open GitHub Actions and show:

- CI workflow: install, ingest, validate, pytest.
- Eval gate workflow: promoted model passes and degraded model blocks locally.

Local command:

```powershell
python -m pytest tests\ -v --tb=short
```

## Minute 8: Monitoring

Show generated monitoring files:

```powershell
python .\monitoring\ml_drift.py
python .\monitoring\rag_monitor.py
```

Open:

- `monitoring/ml_drift_report.html`
- `monitoring/rag_monitoring_report.json`

Close by explaining the next hardening steps: auth, rate limits, PII audit, shadow mode, and A/B routing.

