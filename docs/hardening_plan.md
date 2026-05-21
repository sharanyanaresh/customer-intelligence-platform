# Hardening Plan

| Area | Production Addition | Why It Matters |
|---|---|---|
| API security | Add API-key authentication or OAuth in front of FastAPI. | Public scoring and complaint endpoints should not be unauthenticated in production. |
| Rate limiting | Add per-client limits with Redis-backed counters. | This protects the model and retrieval service from accidental or abusive traffic spikes. |
| Shadow mode | Score new candidate models in parallel without serving their decisions. | It lets the team compare real traffic behavior before promotion. |
| A/B routing | Route a small percentage of traffic to a new promoted model. | This reduces risk when changing campaign decision logic. |
| PII audit | Add automated narrative redaction checks before indexing CFPB text. | Complaint narratives can contain sensitive details, so indexing should have a stronger privacy gate. |
| Observability | Export Prometheus metrics and structured JSON logs. | In-memory metrics are fine for a project, but production needs durable time-series monitoring. |
| Model registry | Use explicit MLflow model stages or aliases. | A real release process needs stable `champion` and `candidate` model references. |

