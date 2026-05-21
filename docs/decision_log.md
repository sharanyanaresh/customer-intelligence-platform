# Decision Log

| Decision | Rejected Option | Chosen Option | Reason |
|---|---|---|---|
| Complaint generation | Paid hosted LLM by default | Local evidence-grounded answer function | It avoids API keys and cost while still enforcing evidence-only behavior. |
| CFPB ingestion | Fail if the public CFPB download is unavailable | Public download with synthetic fallback | The project remains demoable even if the large public file or network is unreliable. |
| Embeddings | Require MiniLM to succeed on every machine | MiniLM first, deterministic hashing fallback | Torch failed to load on the local Python 3.12 setup, so the fallback preserves FAISS retrieval for grading. |
| Model promotion | Absolute metric threshold only | Relative gate versus baseline | A relative gate proves the improved model adds value over a simpler baseline. |
| Serving preprocessing | Recompute encodings from request data | Load scaler and categorical mappings from MLflow artifacts | Train/serve transformation consistency is more important than convenience. |
| Raw data storage | Commit CSV and JSON data | Keep raw data out of Git with hash sidecars | This keeps the repository lightweight and documents data versions without storing large files. |
| API tests | Require live MLflow model in tests | Use a deterministic dummy model bundle | CI should test API behavior without depending on a previous local training run. |

