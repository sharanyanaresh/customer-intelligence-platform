"""Tests for Pydantic request and response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.serving.schemas import BatchScoreRequest, CustomerFeatures


VALID_PAYLOAD = {
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


def test_customer_features_accepts_valid_payload() -> None:
    payload = CustomerFeatures(**VALID_PAYLOAD)

    assert payload.age == 42
    assert payload.default == "no"
    assert payload.pdays == -1
    assert payload.job == "management"


def test_customer_features_rejects_invalid_business_values() -> None:
    invalid = VALID_PAYLOAD | {"age": 12, "duration": 0, "pdays": 0}

    with pytest.raises(ValidationError) as exc:
        CustomerFeatures(**invalid)

    message = str(exc.value)
    assert "age" in message
    assert "duration" in message
    assert "pdays" in message


def test_batch_score_request_requires_one_input_source() -> None:
    valid = BatchScoreRequest(records=[CustomerFeatures(**VALID_PAYLOAD)])

    assert valid.records is not None
    assert valid.csv_file_path is None

    with pytest.raises(ValidationError):
        BatchScoreRequest(records=[CustomerFeatures(**VALID_PAYLOAD)], csv_file_path="data.csv")

    with pytest.raises(ValidationError):
        BatchScoreRequest()

