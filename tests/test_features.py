"""Tests for reusable feature engineering functions."""

from __future__ import annotations

import pandas as pd

from src.data_pipeline.features import (
    build_features,
    bin_age,
    compute_contact_features,
    encode_categoricals,
    get_feature_names,
    scale_numerics,
)


def sample_frame() -> pd.DataFrame:
    """Return a small valid Bank Marketing-like feature frame."""
    return pd.DataFrame(
        [
            {
                "age": 25,
                "job": "admin.",
                "marital": "single",
                "education": "secondary",
                "default": "no",
                "balance": 100,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 10,
                "month": "may",
                "duration": 120,
                "campaign": 1,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown",
            },
            {
                "age": 65,
                "job": "retired",
                "marital": "married",
                "education": "primary",
                "default": "no",
                "balance": 5000,
                "housing": "no",
                "loan": "yes",
                "contact": "telephone",
                "day": 20,
                "month": "jun",
                "duration": 300,
                "campaign": 3,
                "pdays": 7,
                "previous": 2,
                "poutcome": "success",
            },
        ]
    )


def test_encode_categoricals_creates_reusable_integer_columns() -> None:
    encoded, mappings = encode_categoricals(sample_frame())

    assert "job_encoded" in encoded.columns
    assert encoded["job_encoded"].dtype.kind in {"i", "u"}
    assert mappings["job"]["admin."] > 0
    assert mappings["housing"]["yes"] > 0


def test_bin_age_creates_expected_groups() -> None:
    transformed = bin_age(sample_frame())

    assert transformed.loc[0, "age_group"] == "young"
    assert transformed.loc[1, "age_group"] == "senior"
    assert transformed["age_group_encoded"].tolist() == [0, 2]


def test_compute_contact_features_handles_no_previous_contact() -> None:
    transformed = compute_contact_features(sample_frame())

    assert transformed.loc[0, "days_since_contact"] == 999
    assert transformed.loc[1, "days_since_contact"] == 7
    assert transformed.loc[1, "contact_intensity"] == 5


def test_scale_numerics_reuses_prefitted_scaler() -> None:
    df = compute_contact_features(sample_frame())
    scaled_train, scaler = scale_numerics(df)
    scaled_serve, reused_scaler = scale_numerics(df, scaler=scaler)

    assert reused_scaler is scaler
    assert abs(float(scaled_train["balance"].mean())) < 1e-9
    assert scaled_serve["duration"].tolist() == scaled_train["duration"].tolist()


def test_build_features_returns_ordered_model_columns() -> None:
    features, scaler, mappings = build_features(sample_frame())

    assert list(features.columns) == get_feature_names()
    assert features.shape == (2, len(get_feature_names()))
    assert scaler is not None
    assert "job" in mappings

