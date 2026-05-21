"""Evaluate campaign conversion models and enforce the promotion gate."""

from __future__ import annotations

import sys
import time
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data_pipeline.features import build_features


DATA_PATH = PROJECT_ROOT / "data" / "bank_marketing.csv"
RANDOM_STATE = 42
THRESHOLD = 0.4


@dataclass(frozen=True)
class EvaluationResult:
    """Model metrics used by the promotion gate."""

    name: str
    roc_auc: float
    pr_auc: float
    f1: float
    brier_score: float
    latency_ms: float
    confusion_matrix: list[list[int]]


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load local UCI data, build features, and return a deterministic split."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run python src/data_pipeline/ingest.py --sample 5000 first.")

    df = pd.read_csv(DATA_PATH)
    y = df["y"].map({"no": 0, "yes": 1}).astype(int)
    raw_X = df.drop(columns=["y"])
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        raw_X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, scaler, mappings = build_features(X_raw_train)
    X_test, _, _ = build_features(X_raw_test, scaler=scaler, mappings=mappings)
    return X_train, X_test, y_train, y_test


def measure_latency_ms(model, one_row: pd.DataFrame, repeats: int = 50) -> float:
    """Measure average single-row prediction latency."""
    started = time.perf_counter()
    for _ in range(repeats):
        model.predict_proba(one_row)
    return ((time.perf_counter() - started) / repeats) * 1000


def evaluate_model(name: str, model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> EvaluationResult:
    """Fit and evaluate one model with campaign-relevant metrics."""
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)
    matrix = confusion_matrix(y_test, predictions)
    return EvaluationResult(
        name=name,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        pr_auc=float(average_precision_score(y_test, probabilities)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
        brier_score=float(brier_score_loss(y_test, probabilities)),
        latency_ms=float(measure_latency_ms(model, X_test.head(1))),
        confusion_matrix=matrix.astype(int).tolist(),
    )


def check_promotion(baseline: EvaluationResult, improved: EvaluationResult) -> tuple[bool, str]:
    """Return whether the improved model passes the relative promotion gate."""
    pr_auc_delta = improved.pr_auc - baseline.pr_auc
    f1_delta = improved.f1 - baseline.f1
    latency = improved.latency_ms

    failures = []
    if pr_auc_delta < 0.03:
        failures.append(f"PR-AUC delta {pr_auc_delta:.4f} is below required +0.0300")
    if f1_delta < -0.02:
        failures.append(f"F1 delta {f1_delta:.4f} is below allowed -0.0200")
    if latency > 200:
        failures.append(f"latency {latency:.2f}ms exceeds 200ms")

    deltas = f"PR-AUC delta={pr_auc_delta:.4f}, F1 delta={f1_delta:.4f}, latency={latency:.2f}ms"
    if failures:
        return False, f"BLOCKED - reason: {'; '.join(failures)}. {deltas}"
    return True, f"PROMOTED - {deltas}"


def print_result(result: EvaluationResult) -> None:
    """Print one model's metrics in a readable format."""
    print(
        f"{result.name}: ROC-AUC={result.roc_auc:.4f}, PR-AUC={result.pr_auc:.4f}, "
        f"F1={result.f1:.4f}, Brier={result.brier_score:.4f}, "
        f"latency_ms={result.latency_ms:.2f}, confusion_matrix={result.confusion_matrix}"
    )


def print_business_interpretation(baseline: EvaluationResult, improved: EvaluationResult) -> None:
    """Print three concise campaign ROI interpretations."""
    print("Business interpretation:")
    print(f"1. PR-AUC improved from {baseline.pr_auc:.4f} to {improved.pr_auc:.4f}, which matters because subscribed customers are the minority class.")
    print(f"2. F1 at threshold {THRESHOLD:.1f} balances missed converters against wasted calls, helping campaign teams focus outreach effort.")
    print(f"3. Brier score {improved.brier_score:.4f} indicates probability calibration quality, which affects budget allocation by conversion band.")


def build_models(degraded: bool = False) -> tuple[LogisticRegression, XGBClassifier]:
    """Create baseline and improved/degraded models."""
    baseline = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    improved = XGBClassifier(
        n_estimators=5 if degraded else 200,
        max_depth=2 if degraded else 5,
        learning_rate=0.01 if degraded else 0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return baseline, improved


def run_gate_demo(degraded: bool = False) -> bool:
    """Train two models and print the promotion decision."""
    X_train, X_test, y_train, y_test = load_split()
    baseline_model, improved_model = build_models(degraded=degraded)
    baseline = evaluate_model("baseline", baseline_model, X_train, X_test, y_train, y_test)
    improved_name = "improved_degraded" if degraded else "improved"
    improved = evaluate_model(improved_name, improved_model, X_train, X_test, y_train, y_test)

    print_result(baseline)
    print_result(improved)
    print_business_interpretation(baseline, improved)
    promoted, message = check_promotion(baseline, improved)
    icon = "✅" if promoted else "🚫"
    print(f"{icon} {message}")
    return promoted


def main() -> None:
    """Run either the CI gate or both demo outcomes."""
    parser = argparse.ArgumentParser(description="Evaluate model promotion gate.")
    parser.add_argument("--mode", choices=["demo", "gate"], default="demo")
    args = parser.parse_args()

    if args.mode == "gate":
        promoted = run_gate_demo(degraded=False)
        if not promoted:
            sys.exit(1)
        return

    """Demonstrate both a passing and blocked promotion-gate outcome."""
    print("Promotion gate demo 1: expected PASS")
    pass_result = run_gate_demo(degraded=False)
    print()
    print("Promotion gate demo 2: intentional degraded model expected BLOCK")
    block_result = run_gate_demo(degraded=True)

    if not pass_result:
        print("Warning: the non-degraded improved model did not pass on this local split.")
    if block_result:
        print("Warning: the degraded model unexpectedly passed; inspect gate thresholds.")


if __name__ == "__main__":
    main()
