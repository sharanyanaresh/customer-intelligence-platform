"""Train baseline and improved campaign conversion models with MLflow tracking."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.features import build_features, get_feature_names
from src.mlflow_config import configure_mlflow


DATA_PATH = PROJECT_ROOT / "data" / "bank_marketing.csv"
HASH_PATH = PROJECT_ROOT / "data" / "uci_hash.txt"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "training_artifacts"
RANDOM_STATE = 42
CHOSEN_THRESHOLD = 0.4


def load_dataset() -> tuple[pd.DataFrame, pd.Series, str]:
    """Load the validated UCI dataset and return features, target, and dataset hash."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run src/data_pipeline/ingest.py first.")
    df = pd.read_csv(DATA_PATH)
    dataset_hash = HASH_PATH.read_text(encoding="utf-8").split()[0] if HASH_PATH.exists() else "unknown"
    target = df["y"].map({"no": 0, "yes": 1}).astype(int)
    features = df.drop(columns=["y"])
    return features, target, dataset_hash


def compute_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute campaign model metrics at the chosen threshold."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": threshold,
        "confusion_tn": float(tn),
        "confusion_fp": float(fp),
        "confusion_fn": float(fn),
        "confusion_tp": float(tp),
    }


def threshold_analysis(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    """Return precision and recall for standard campaign decision thresholds."""
    results: dict[str, float] = {}
    for threshold in [0.3, 0.4, 0.5]:
        predictions = (probabilities >= threshold).astype(int)
        results[f"precision_at_{threshold}"] = float(precision_score(y_true, predictions, zero_division=0))
        results[f"recall_at_{threshold}"] = float(recall_score(y_true, predictions, zero_division=0))
    return results


def measure_latency_ms(model, sample: pd.DataFrame, repeats: int = 25) -> float:
    """Measure average one-row inference latency in milliseconds."""
    row = sample.head(1)
    started = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(row)
    return float(((time.perf_counter() - started) / repeats) * 1000)


def save_json_artifact(payload: dict, path: Path) -> Path:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def log_common_artifacts(model, scaler, mappings: dict, run_name: str, X_test: pd.DataFrame) -> None:
    """Log model, scaler, mappings, and feature names to MLflow."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = ARTIFACT_DIR / f"{run_name}_scaler.joblib"
    mappings_path = ARTIFACT_DIR / f"{run_name}_categorical_mappings.json"
    features_path = ARTIFACT_DIR / f"{run_name}_feature_names.json"

    joblib.dump(scaler, scaler_path)
    save_json_artifact(mappings, mappings_path)
    save_json_artifact({"feature_names": get_feature_names()}, features_path)

    mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_test.head(1))
    mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")
    mlflow.log_artifact(str(mappings_path), artifact_path="preprocessing")
    mlflow.log_artifact(str(features_path), artifact_path="preprocessing")


def log_feature_importance(model: XGBClassifier, feature_names: list[str]) -> None:
    """Log a feature importance bar chart for the improved model."""
    importances = model.feature_importances_
    order = np.argsort(importances)[-12:]
    plt.figure(figsize=(9, 5))
    plt.barh(np.array(feature_names)[order], importances[order])
    plt.title("Top XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    output_path = ARTIFACT_DIR / "improved_feature_importance.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    mlflow.log_artifact(str(output_path), artifact_path="plots")


def log_calibration_curve(y_true: pd.Series, probabilities: np.ndarray) -> None:
    """Log a probability calibration curve for the improved model."""
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Improved model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.title("Calibration Curve")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.tight_layout()
    output_path = ARTIFACT_DIR / "improved_calibration_curve.png"
    plt.savefig(output_path, dpi=160)
    plt.close()
    mlflow.log_artifact(str(output_path), artifact_path="plots")


def train_and_log_model(
    run_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler,
    mappings: dict,
    dataset_hash: str,
) -> dict[str, float | str]:
    """Fit one model, log parameters/metrics/artifacts, and return the run summary."""
    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, probabilities, CHOSEN_THRESHOLD)
        metrics.update(threshold_analysis(y_test, probabilities))
        metrics["inference_latency_ms"] = measure_latency_ms(model, X_test)

        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_param("chosen_threshold", CHOSEN_THRESHOLD)
        mlflow.log_param("feature_count", len(get_feature_names()))
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        log_common_artifacts(model, scaler, mappings, run_name, X_test)

        if run_name == "improved":
            log_feature_importance(model, get_feature_names())
            log_calibration_curve(y_test, probabilities)

        print(
            f"{run_name} | run_id={run.info.run_id} | "
            f"roc_auc={metrics['roc_auc']:.4f} | pr_auc={metrics['pr_auc']:.4f} | "
            f"f1={metrics['f1']:.4f} | latency_ms={metrics['inference_latency_ms']:.2f}"
        )
        return {"run_id": run.info.run_id, **metrics}


def main() -> None:
    """Train baseline LogisticRegression and improved XGBClassifier models."""
    configure_mlflow()
    raw_features, target, dataset_hash = load_dataset()
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        raw_features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    X_train, scaler, mappings = build_features(X_raw_train)
    X_test, _, _ = build_features(X_raw_test, scaler=scaler, mappings=mappings)

    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / max(positive_count, 1)

    baseline_model = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    improved_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training baseline and improved models with MLflow tracking...")
    baseline = train_and_log_model("baseline", baseline_model, X_train, X_test, y_train, y_test, scaler, mappings, dataset_hash)
    improved = train_and_log_model("improved", improved_model, X_train, X_test, y_train, y_test, scaler, mappings, dataset_hash)
    print(
        "Training complete. "
        f"PR-AUC delta={improved['pr_auc'] - baseline['pr_auc']:.4f}, "
        f"F1 delta={improved['f1'] - baseline['f1']:.4f}."
    )


if __name__ == "__main__":
    main()
