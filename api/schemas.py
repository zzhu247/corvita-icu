"""Request and response shapes.

Rule of thumb for what is rejected (422) vs accepted with a warning:
  - A request the client built wrong -> 422. Wrong types, unknown fields, unknown
    vital names, bad time strings, age outside the training range, ICU type not 1-4.
  - Clinical data that is merely incomplete or noisy -> 200 with warnings, because
    the same thing happened in training and the model was built to handle it:
    vitals after 24:00 (ignored), implausible values (ignored), vitals not measured.
"""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import load_config

CFG = load_config()
AGE_MIN, AGE_MAX = CFG["valid_ranges"]["Age"]
TIME_RE = re.compile(r"^(\d{1,3}):([0-5]\d)$")


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: str = Field(description="HH:MM since ICU admission, e.g. '07:30' or '23:59'")
    parameter: Literal["HR", "RespRate", "Temp"]
    value: float = Field(allow_inf_nan=False)

    @field_validator("time")
    @classmethod
    def check_time(cls, v: str) -> str:
        if not TIME_RE.match(v):
            raise ValueError("time must be HH:MM with minutes 00-59, e.g. '07:30'")
        return v


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str | None = Field(default=None, max_length=64,
                                  description="Echoed back, never logged")
    age: float = Field(ge=AGE_MIN, le=AGE_MAX, description="Years. Training data is adults only.")
    gender: Literal[0, 1] | None = Field(default=None, description="0 female, 1 male, null unknown")
    icu_type: Literal[1, 2, 3, 4] = Field(
        description="1 coronary, 2 cardiac surgery recovery, 3 medical, 4 surgical")
    observations: list[Observation] = Field(default_factory=list, max_length=5000)


class PredictResponse(BaseModel):
    record_id: str | None
    probability: float
    high_risk: bool
    threshold: float
    model_version: str
    data_version: str
    data_quality_flag: Literal["ok", "missing_vitals", "insufficient_data"]
    missing_vitals: list[str]
    warnings: list[str]
    request_id: str
