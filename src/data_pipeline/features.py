"""Reusable feature engineering for Bank Marketing conversion prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.preprocessing import StandardScaler


CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

NUMERIC_COLUMNS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "days_since_contact",
    "contact_intensity",
]

FINAL_FEATURE_COLUMNS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "job_encoded",
    "marital_encoded",
    "education_encoded",
    "default_encoded",
    "housing_encoded",
    "loan_encoded",
    "contact_encoded",
    "month_encoded",
    "poutcome_encoded",
    "age_group_encoded",
    "days_since_contact",
    "contact_intensity",
]

AGE_GROUP_ORDER = {"young": 0, "mid": 1, "senior": 2}


@dataclass(frozen=True)
class FeatureTransformResult:
    """Container for transformed data and fitted reusable transformers."""

    dataframe: pd.DataFrame
    scaler: StandardScaler


def _stable_category_mapping(values: Iterable[object]) -> dict[str, int]:
    """Build a deterministic label mapping with unknown values reserved as 0."""
    normalized = sorted({str(value).strip().lower() for value in values if pd.notna(value)})
    return {"__unknown__": 0, **{value: index + 1 for index, value in enumerate(normalized)}}


def encode_categoricals(
    df: pd.DataFrame,
    mappings: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Label-encode known categorical columns with reusable deterministic mappings."""
    encoded = df.copy()
    fitted_mappings = {} if mappings is None else {column: mapping.copy() for column, mapping in mappings.items()}

    for column in CATEGORICAL_COLUMNS:
        if column not in encoded.columns:
            raise KeyError(f"Missing categorical column: {column}")
        if column not in fitted_mappings:
            fitted_mappings[column] = _stable_category_mapping(encoded[column])
        normalized = encoded[column].astype(str).str.strip().str.lower()
        encoded[f"{column}_encoded"] = normalized.map(fitted_mappings[column]).fillna(0).astype(int)

    return encoded, fitted_mappings


def bin_age(df: pd.DataFrame) -> pd.DataFrame:
    """Create age_group and age_group_encoded using business-friendly age bands."""
    if "age" not in df.columns:
        raise KeyError("Missing required column: age")
    transformed = df.copy()
    transformed["age_group"] = pd.cut(
        transformed["age"],
        bins=[17, 30, 60, 95],
        labels=["young", "mid", "senior"],
        include_lowest=True,
    ).astype(str)
    transformed["age_group_encoded"] = transformed["age_group"].map(AGE_GROUP_ORDER).fillna(0).astype(int)
    return transformed


def compute_contact_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create contact recency and intensity features from campaign history fields."""
    required = {"pdays", "campaign", "previous"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required contact columns: {sorted(missing)}")

    transformed = df.copy()
    transformed["days_since_contact"] = transformed["pdays"].where(transformed["pdays"] > 0, 999)
    transformed["contact_intensity"] = transformed["campaign"] + transformed["previous"]
    return transformed


def scale_numerics(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Fit or apply StandardScaler to numeric columns for train/serve consistency."""
    missing = [column for column in NUMERIC_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing numeric columns for scaling: {missing}")

    transformed = df.copy()
    active_scaler = scaler or StandardScaler()
    if scaler is None:
        transformed[NUMERIC_COLUMNS] = active_scaler.fit_transform(transformed[NUMERIC_COLUMNS])
    else:
        transformed[NUMERIC_COLUMNS] = active_scaler.transform(transformed[NUMERIC_COLUMNS])
    return transformed, active_scaler


def get_feature_names() -> list[str]:
    """Return the ordered model feature columns used by train and serve."""
    return FINAL_FEATURE_COLUMNS.copy()


def build_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    mappings: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, StandardScaler, dict[str, dict[str, int]]]:
    """Run the full feature pipeline and return model-ready features."""
    transformed = bin_age(df)
    transformed = compute_contact_features(transformed)
    transformed, fitted_mappings = encode_categoricals(transformed, mappings=mappings)
    transformed, fitted_scaler = scale_numerics(transformed, scaler=scaler)
    return transformed[get_feature_names()].copy(), fitted_scaler, fitted_mappings

