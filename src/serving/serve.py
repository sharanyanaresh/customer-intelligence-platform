"""FastAPI service for ML campaign scoring and operational metrics."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.features import build_features
from src.serving.model_loader import LoadedModel, load_serving_bundle
from src.serving.schemas import (
    BatchScoreItem,
    BatchScoreRequest,
    BatchScoreResponse,
    CustomerIntelRequest,
    CustomerIntelResponse,
    CustomerFeatures,
    MetricsResponse,
    PredictionResponse,
)


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.4"))
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("customer-intel-api")

APP_STATE: dict[str, Any] = {
    "bundle": None,
    "healthy": False,
    "started_at": time.time(),
    "latencies": [],
    "request_count": 0,
    "error_count": 0,
    "prediction_distribution": Counter({"low": 0, "medium": 0, "high": 0}),
}


def conversion_band(probability: float) -> str:
    """Convert a probability into the requested business band."""
    if probability < 0.3:
        return "low"
    if probability <= 0.6:
        return "medium"
    return "high"


def _bundle() -> LoadedModel:
    """Return the loaded model bundle or raise a service error."""
    bundle = APP_STATE.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return bundle


def _features_to_frame(records: list[CustomerFeatures]) -> pd.DataFrame:
    """Convert validated request objects to a pandas DataFrame."""
    return pd.DataFrame([record.model_dump() for record in records])


def predict_records(records: list[CustomerFeatures]) -> tuple[np.ndarray, float, str]:
    """Run model inference for one or more validated records."""
    bundle = _bundle()
    started = time.perf_counter()
    raw = _features_to_frame(records)
    features, _, _ = build_features(raw, scaler=bundle.scaler, mappings=bundle.mappings)
    probabilities = bundle.model.predict_proba(features)[:, 1]
    latency_ms = (time.perf_counter() - started) * 1000
    return probabilities, latency_ms, bundle.run_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the serving bundle on startup and expose health status."""
    try:
        APP_STATE["bundle"] = load_serving_bundle()
        APP_STATE["healthy"] = True
        logger.info("Loaded model run_id=%s", APP_STATE["bundle"].run_id)
    except Exception:
        APP_STATE["healthy"] = False
        logger.exception("Failed to load model bundle")
    yield


app = FastAPI(title="Customer Intelligence Platform", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    """Log every request with method, path, status, and latency."""
    started = time.perf_counter()
    APP_STATE["request_count"] += 1
    try:
        response = await call_next(request)
    except Exception:
        APP_STATE["error_count"] += 1
        logger.exception("Unhandled request error")
        raise
    latency_ms = (time.perf_counter() - started) * 1000
    APP_STATE["latencies"].append(latency_ms)
    if response.status_code >= 400:
        APP_STATE["error_count"] += 1
    logger.info("%s %s status=%s latency_ms=%.2f", request.method, request.url.path, response.status_code, latency_ms)
    return response


@app.get("/health")
def health() -> dict[str, str | float]:
    """Return service health, model version, index version, and uptime."""
    bundle = APP_STATE.get("bundle")
    return {
        "status": "ok" if APP_STATE["healthy"] else "degraded",
        "model_version": bundle.run_id if bundle else "not-loaded",
        "index_version": os.getenv("FAISS_INDEX_PATH", "data/faiss.index"),
        "uptime_seconds": round(time.time() - APP_STATE["started_at"], 3),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerFeatures) -> PredictionResponse:
    """Score one customer for campaign conversion."""
    probabilities, latency_ms, run_id = predict_records([payload])
    probability = float(probabilities[0])
    band = conversion_band(probability)
    APP_STATE["prediction_distribution"][band] += 1
    return PredictionResponse(
        prediction=int(probability >= THRESHOLD),
        probability=probability,
        threshold_decision=bool(probability >= THRESHOLD),
        model_version=run_id,
        latency_ms=latency_ms,
    )


@app.post("/batch-score", response_model=BatchScoreResponse)
def batch_score(payload: BatchScoreRequest) -> BatchScoreResponse:
    """Score a list of customers or a local CSV path."""
    if payload.records is not None:
        records = payload.records
    else:
        csv_path = Path(str(payload.csv_file_path))
        if not csv_path.exists():
            raise HTTPException(status_code=400, detail=f"CSV file not found: {csv_path}")
        records = [CustomerFeatures(**row) for row in pd.read_csv(csv_path).to_dict(orient="records")]

    probabilities, _, _ = predict_records(records)
    results: list[BatchScoreItem] = []
    for index, probability in enumerate(probabilities):
        probability_float = float(probability)
        band = conversion_band(probability_float)
        APP_STATE["prediction_distribution"][band] += 1
        results.append(BatchScoreItem(id=index, conversion_band=band, probability=probability_float))
    return BatchScoreResponse(results=results)


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Return lightweight in-memory API metrics."""
    latencies = np.array(APP_STATE["latencies"], dtype=float)
    p50 = float(np.percentile(latencies, 50)) if len(latencies) else 0.0
    p99 = float(np.percentile(latencies, 99)) if len(latencies) else 0.0
    return MetricsResponse(
        latency_p50=p50,
        latency_p99=p99,
        request_count=int(APP_STATE["request_count"]),
        error_count=int(APP_STATE["error_count"]),
        prediction_distribution=dict(APP_STATE["prediction_distribution"]),
    )


@app.post("/customer-intel", response_model=CustomerIntelResponse)
def customer_intel(payload: CustomerIntelRequest) -> CustomerIntelResponse:
    """Combine conversion scoring with complaint themes and cited records."""
    probabilities, _, run_id = predict_records([payload.customer])
    probability = float(probabilities[0])
    band = conversion_band(probability)
    APP_STATE["prediction_distribution"][band] += 1

    try:
        from src.rag.retrieve import retrieve
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Complaint retriever unavailable: {exc}") from exc

    filters = {
        key: value
        for key, value in {
            "product": payload.product,
            "issue": payload.issue,
            "date_from": payload.date_from,
            "date_to": payload.date_to,
        }.items()
        if value
    }
    question = "What are the top complaint themes for these customer complaint filters?"
    chunks, refused = retrieve(question, top_k=5, filters=filters, threshold=0.15)

    themes: list[str] = []
    cited_record_ids: list[str] = []
    if not refused:
        for chunk in chunks:
            issue = str(chunk.get("issue", "Unknown issue"))
            complaint_id = str(chunk.get("complaint_id", ""))
            if issue not in themes:
                themes.append(issue)
            if complaint_id and complaint_id not in cited_record_ids:
                cited_record_ids.append(complaint_id)

    return CustomerIntelResponse(
        conversion_band=band,
        conversion_probability=probability,
        top_complaint_themes=themes[:5],
        cited_record_ids=cited_record_ids[:5],
        model_version=run_id,
        index_version=os.getenv("FAISS_INDEX_PATH", "data/faiss.index"),
    )
