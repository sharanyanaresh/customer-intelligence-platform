"""Validate the UCI Bank Marketing dataset with Pandera."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "bank_marketing.csv"


UCI_SCHEMA = DataFrameSchema(
    {
        "age": Column(int, checks=Check.in_range(18, 95), nullable=False),
        "job": Column(str, nullable=False),
        "marital": Column(str, nullable=False),
        "education": Column(str, nullable=False),
        "default": Column(str, checks=Check.isin(["yes", "no"]), nullable=False),
        "balance": Column(int, nullable=False),
        "housing": Column(str, checks=Check.isin(["yes", "no"]), nullable=False),
        "loan": Column(str, checks=Check.isin(["yes", "no"]), nullable=False),
        "contact": Column(str, nullable=False),
        "day": Column(int, checks=Check.in_range(1, 31), nullable=False),
        "month": Column(str, nullable=False),
        "duration": Column(int, checks=Check.greater_than(0), nullable=False),
        "campaign": Column(int, checks=Check.greater_than(0), nullable=False),
        "pdays": Column(int, checks=Check(lambda series: ((series == -1) | (series > 0)).all(), element_wise=False), nullable=False),
        "previous": Column(int, checks=Check.greater_than_or_equal_to(0), nullable=False),
        "poutcome": Column(str, nullable=False),
        "y": Column(str, checks=Check.isin(["yes", "no"]), nullable=False),
    },
    strict=True,
    coerce=True,
)


def validate_uci(path: Path = DATA_PATH) -> pd.DataFrame:
    """Validate UCI data and return the validated frame."""
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}. Run ingest.py first.")
    df = pd.read_csv(path)
    return UCI_SCHEMA.validate(df, lazy=True)


def main() -> None:
    """Validate the local UCI CSV and print a clear pass/fail result."""
    try:
        validated = validate_uci()
    except pa.errors.SchemaErrors as exc:
        print("Validation failed. Failing rows/checks:")
        print(exc.failure_cases.to_string(index=False))
        sys.exit(1)
    except Exception as exc:
        print(f"Validation failed before schema checks: {exc}")
        sys.exit(1)

    rows, columns = validated.shape
    print(f"Validation passed - {rows} rows, {columns} columns.")


if __name__ == "__main__":
    main()
