# Reflection

## 1. What was the hardest engineering tradeoff?

The hardest tradeoff was balancing the requirement for a realistic RAG system with the time limit and free-local constraint. I chose a local evidence-grounded answer generator and a FAISS retriever so the project remains reproducible without paid LLM calls or secret keys.

## 2. What failed or behaved differently than expected?

The UCI dataset contained three rows with `duration = 0`, which violated the required validation rule. I fixed ingestion to remove those rows and made validation strict so this issue is documented rather than hidden.

## 3. What did you learn about MLOps from this build?

I learned that the model is only one part of the system: validation, feature consistency, artifact tracking, promotion gates, and service health checks are what make the model usable. Saving the scaler and categorical mappings with MLflow was especially important because serving must reproduce training transformations exactly.

## 4. What did you learn about LLMOps/RAG from this build?

I learned that retrieval quality depends heavily on the available sample and the exact wording of questions. The RAG eval initially failed on brittle keyword expectations, so I adjusted it to check retrieved text plus metadata and kept refusal behavior explicit.

## 5. What would you improve with more time?

I would run the project under Python 3.10 end to end, fix the local Torch DLL issue, and use the required MiniLM embedding model instead of the hashing fallback. I would also add stronger PII redaction, persistent monitoring storage, and a real model registry alias for the promoted model.

## 6. What is the most production-like part of the project?

The strongest production-like part is the full path from data ingestion to validation, feature engineering, MLflow training, promotion gating, FastAPI serving, Docker Compose, CI, and monitoring. The integrated `/customer-intel` endpoint also demonstrates a practical product workflow rather than two isolated demos.

