"""Generate an ML drift report for the Bank Marketing dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "bank_marketing.csv"
REPORT_PATH = PROJECT_ROOT / "monitoring" / "ml_drift_report.html"


def load_reference() -> pd.DataFrame:
    """Load the validated UCI reference dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run python src/data_pipeline/ingest.py --sample 5000 first.")
    return pd.read_csv(DATA_PATH)


def create_drifted_dataset(reference: pd.DataFrame) -> pd.DataFrame:
    """Create a synthetic drifted copy using the requested shifts."""
    drifted = reference.copy()
    rng = np.random.default_rng(42)

    age_indices = rng.choice(drifted.index, size=int(len(drifted) * 0.30), replace=False)
    drifted.loc[age_indices, "age"] = (drifted.loc[age_indices, "age"] + 15).clip(upper=95)

    outcome_indices = rng.choice(drifted.index, size=int(len(drifted) * 0.20), replace=False)
    drifted.loc[outcome_indices, "y"] = drifted.loc[outcome_indices, "y"].map({"yes": "no", "no": "yes"})

    missing_indices = rng.choice(drifted.index, size=int(len(drifted) * 0.05), replace=False)
    drifted.loc[missing_indices, "balance"] = np.nan
    return drifted


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute a simple population stability index for numeric columns."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_summary(reference: pd.DataFrame, drifted: pd.DataFrame) -> tuple[list[str], float, float]:
    """Return drifted features, max PSI, and an overall drift score."""
    numeric_columns = reference.select_dtypes(include=["number"]).columns
    psi_by_column = {column: psi(reference[column], drifted[column]) for column in numeric_columns}
    missing_delta = abs(drifted["balance"].isna().mean() - reference["balance"].isna().mean())
    outcome_delta = abs(drifted["y"].value_counts(normalize=True).get("yes", 0) - reference["y"].value_counts(normalize=True).get("yes", 0))
    psi_by_column["balance_missing_rate"] = float(missing_delta)
    psi_by_column["target_yes_rate"] = float(outcome_delta)
    drifted_features = [column for column, value in psi_by_column.items() if value > 0.05]
    max_psi = max(psi_by_column.values()) if psi_by_column else 0.0
    overall_drift_score = float(np.mean([min(value, 1.0) for value in psi_by_column.values()]))
    return drifted_features, max_psi, overall_drift_score


def write_evidently_report(reference: pd.DataFrame, drifted: pd.DataFrame) -> bool:
    """Try to write an Evidently HTML report using the installed 0.4 API."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=drifted)
        report.save_html(str(REPORT_PATH))
        return True
    except Exception as exc:
        print(f"Evidently report generation unavailable, writing fallback HTML. Reason: {exc}")
        return False


def write_fallback_report(reference: pd.DataFrame, drifted: pd.DataFrame, drifted_features: list[str], max_psi: float, overall: float) -> None:
    """Write a simple HTML fallback report if Evidently cannot render."""
    REPORT_PATH.write_text(
        f"""<!doctype html>
<html><head><title>ML Drift Report</title></head>
<body>
<h1>ML Drift Report</h1>
<p><strong>Requested report types:</strong> ColumnDriftReport + DatasetDriftReport equivalent summary.</p>
<p><strong>Reference rows:</strong> {len(reference)} | <strong>Drifted rows:</strong> {len(drifted)}</p>
<p><strong>Drifted features:</strong> {', '.join(drifted_features)}</p>
<p><strong>Max PSI:</strong> {max_psi:.4f}</p>
<p><strong>Overall drift score:</strong> {overall:.4f}</p>
</body></html>
""",
        encoding="utf-8",
    )


def main() -> None:
    """Generate and summarize the ML drift report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference = load_reference()
    drifted = create_drifted_dataset(reference)
    drifted_features, max_psi, overall = drift_summary(reference, drifted)
    if not write_evidently_report(reference, drifted):
        write_fallback_report(reference, drifted, drifted_features, max_psi, overall)

    print(f"Drifted features: {drifted_features}, max PSI: {max_psi:.4f}")
    if overall > 0.3:
        print(f"⚠️ RETRAIN TRIGGERED - drift score: {overall:.4f}. Action: re-run train.py")
    else:
        print(f"Retrain not triggered - drift score: {overall:.4f}.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()

