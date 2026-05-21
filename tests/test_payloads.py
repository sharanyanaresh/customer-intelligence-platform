"""FastAPI payload tests using a lightweight in-memory model bundle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fastapi.testclient import TestClient

from src.data_pipeline.features import build_features
from src.serving import serve
from tests.test_schemas import VALID_PAYLOAD


class DummyModel:
    """Deterministic classifier used to avoid MLflow dependency in unit tests."""

    def predict_proba(self, features):
        probabilities = np.full(len(features), 0.42)
        return np.column_stack([1 - probabilities, probabilities])


@dataclass
class DummyBundle:
    """Minimal serving bundle shape used by serve.py."""

    model: DummyModel
    scaler: object
    mappings: dict[str, dict[str, int]]
    run_id: str = "test-run"
    loaded_at: str = "2026-05-21T00:00:00+00:00"


def install_dummy_bundle() -> None:
    """Install a deterministic model bundle into app state."""
    _, scaler, mappings = build_features(__import__("pandas").DataFrame([VALID_PAYLOAD, VALID_PAYLOAD | {"age": 55, "balance": 3000}]))
    serve.APP_STATE["bundle"] = DummyBundle(model=DummyModel(), scaler=scaler, mappings=mappings)
    serve.APP_STATE["healthy"] = True
    serve.APP_STATE["latencies"] = []
    serve.APP_STATE["request_count"] = 0
    serve.APP_STATE["error_count"] = 0
    serve.APP_STATE["prediction_distribution"].clear()


def test_predict_endpoint_returns_probability_and_version() -> None:
    install_dummy_bundle()
    client = TestClient(serve.app)
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()

    assert response.status_code == 200
    assert data["probability"] == 0.42
    assert data["model_version"] == "test-run"
    assert data["threshold_decision"] is True


def test_predict_endpoint_rejects_invalid_payload() -> None:
    install_dummy_bundle()
    client = TestClient(serve.app)
    response = client.post("/predict", json={"age": 42, "job": "management"})

    assert response.status_code == 422
    assert "detail" in response.json()
    assert len(response.json()["detail"]) >= 1


def test_batch_score_endpoint_returns_three_results() -> None:
    install_dummy_bundle()
    client = TestClient(serve.app)
    response = client.post("/batch-score", json={"records": [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD]})
    data = response.json()

    assert response.status_code == 200
    assert len(data["results"]) == 3
    assert data["results"][0]["conversion_band"] == "medium"
    assert data["results"][2]["probability"] == 0.42

