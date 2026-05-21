"""Shared MLflow configuration for training, evaluation, and serving."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow


EXPERIMENT_NAME = "customer-intelligence-platform"
ARTIFACT_ROOT = Path(os.getenv("MLFLOW_ARTIFACT_ROOT", "mlruns")).resolve()


def configure_mlflow() -> str:
    """Configure MLflow tracking and return the active experiment name."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return EXPERIMENT_NAME


def get_artifact_root() -> Path:
    """Return the local artifact root used for filesystem-backed MLflow runs."""
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_ROOT


def get_active_run_id() -> str:
    """Return the active MLflow run ID, or raise a clear error if none exists."""
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run. Start mlflow.start_run() before requesting a run ID.")
    return active_run.info.run_id

