"""Pydantic v2 schemas for the Customer Intelligence FastAPI service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CustomerFeatures(BaseModel):
    """All UCI Bank Marketing input fields required for one prediction."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=18, le=95)
    job: str = Field(..., min_length=1)
    marital: str = Field(..., min_length=1)
    education: str = Field(..., min_length=1)
    default: Literal["yes", "no"]
    balance: int
    housing: Literal["yes", "no"]
    loan: Literal["yes", "no"]
    contact: str = Field(..., min_length=1)
    day: int = Field(..., ge=1, le=31)
    month: str = Field(..., min_length=3, max_length=3)
    duration: int = Field(..., gt=0)
    campaign: int = Field(..., gt=0)
    pdays: int
    previous: int = Field(..., ge=0)
    poutcome: str = Field(..., min_length=1)

    @field_validator("pdays")
    @classmethod
    def pdays_must_be_minus_one_or_positive(cls, value: int) -> int:
        """Allow -1 for no previous contact, otherwise require positive days."""
        if value == -1 or value > 0:
            return value
        raise ValueError("pdays must be -1 or a positive integer")

    @field_validator("job", "marital", "education", "contact", "month", "poutcome")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        """Trim whitespace from categorical values."""
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("categorical fields cannot be blank")
        return normalized


class PredictionResponse(BaseModel):
    """Response for one customer prediction."""

    prediction: int
    probability: float
    threshold_decision: bool
    model_version: str
    latency_ms: float


class BatchScoreRequest(BaseModel):
    """Batch scoring input as records or a local CSV file path."""

    records: list[CustomerFeatures] | None = None
    csv_file_path: str | None = None

    @model_validator(mode="after")
    def exactly_one_input_source(self) -> "BatchScoreRequest":
        """Require either JSON records or a CSV path, but not both."""
        has_records = bool(self.records)
        has_csv = bool(self.csv_file_path)
        if has_records == has_csv:
            raise ValueError("Provide exactly one of records or csv_file_path.")
        return self


class BatchScoreItem(BaseModel):
    """One batch-scoring result row."""

    id: int
    conversion_band: Literal["low", "medium", "high"]
    probability: float


class BatchScoreResponse(BaseModel):
    """Response for batch scoring."""

    results: list[BatchScoreItem]


class MetricsResponse(BaseModel):
    """In-memory operational metrics for the API."""

    latency_p50: float
    latency_p99: float
    request_count: int
    error_count: int
    prediction_distribution: dict[str, int]


class CustomerIntelRequest(BaseModel):
    """Integrated request combining customer features and complaint filters."""

    model_config = ConfigDict(extra="forbid")

    customer: CustomerFeatures
    product: str | None = None
    issue: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class CustomerIntelResponse(BaseModel):
    """Integrated ML + complaint intelligence response."""

    conversion_band: Literal["low", "medium", "high"]
    conversion_probability: float
    top_complaint_themes: list[str]
    cited_record_ids: list[str]
    model_version: str
    index_version: str
