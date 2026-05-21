"""Versioned MLflow model loading helpers for FastAPI serving."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.features import build_features
from src.mlflow_config import configure_mlflow


@dataclass(frozen=True)
class LoadedModel:
    """Loaded serving bundle."""

    model: object
    scaler: object
    mappings: dict[str, dict[str, int]]
    run_id: str
    loaded_at: str


def _find_latest_improved_run_id() -> str:
    """Find the newest successful improved MLflow run."""
    configure_mlflow()
    runs = mlflow.search_runs(
        filter_string="tags.mlflow.runName = 'improved'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No improved MLflow run found. Run python src/training/train.py first.")
    return str(runs.iloc[0]["run_id"])


def load_model(run_id: str | None = None):
    """Load a model from MLflow by run_id, or the latest improved run if omitted."""
    configure_mlflow()
    active_run_id = run_id or os.getenv("MODEL_RUN_ID") or _find_latest_improved_run_id()
    model = mlflow.sklearn.load_model(f"runs:/{active_run_id}/model")
    return model, active_run_id


def _download_preprocessing_dir(run_id: str) -> Path:
    """Download or resolve preprocessing artifacts for a run."""
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="preprocessing",
    )
    return Path(path)


def load_scaler(run_id: str):
    """Load the scaler artifact from the same MLflow run as the model."""
    preprocessing_dir = _download_preprocessing_dir(run_id)
    scaler_files = list(preprocessing_dir.glob("*_scaler.joblib"))
    if not scaler_files:
        raise FileNotFoundError(f"No scaler artifact found for run {run_id}.")
    return joblib.load(scaler_files[0])


def load_mappings(run_id: str) -> dict[str, dict[str, int]]:
    """Load categorical mappings from the same MLflow run as the model."""
    preprocessing_dir = _download_preprocessing_dir(run_id)
    mapping_files = list(preprocessing_dir.glob("*_categorical_mappings.json"))
    if not mapping_files:
        raise FileNotFoundError(f"No categorical mappings artifact found for run {run_id}.")
    return json.loads(mapping_files[0].read_text(encoding="utf-8"))


def warm_up(model, scaler, mappings: dict[str, dict[str, int]]) -> None:
    """Run one dummy prediction so the first live request is not cold."""
    dummy = pd.DataFrame(
        [
            {
                "age": 42,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 1200,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 15,
                "month": "may",
                "duration": 180,
                "campaign": 2,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown",
            }
        ]
    )
    features, _, _ = build_features(dummy, scaler=scaler, mappings=mappings)
    model.predict_proba(features)


def load_serving_bundle(run_id: str | None = None) -> LoadedModel:
    """Load model, scaler, mappings, run ID, and timestamp for serving."""
    model, active_run_id = load_model(run_id=run_id)
    scaler = load_scaler(active_run_id)
    mappings = load_mappings(active_run_id)
    warm_up(model, scaler, mappings)
    return LoadedModel(
        model=model,
        scaler=scaler,
        mappings=mappings,
        run_id=active_run_id,
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )
