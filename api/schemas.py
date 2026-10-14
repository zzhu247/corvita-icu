"""Request and response shapes.

What is rejected (422) vs accepted with a warning:
  - A request the client built wrong -> 422. Wrong types (strict: no "60" for a number,
    no true/false or 3.0 for an integer code), unknown fields, unknown vital names,
    bad time strings, non-finite values, age outside the training range, ICU type not 1-4.
  - Clinical data that is merely incomplete or noisy -> 200 with warnings, because
    the same thing happened in training and the model was built to handle it:
    vitals after 24:00 (ignored), implausible values (ignored), vitals not measured.

Input contract: send every charted reading from the first 24 h, not summaries.
The model uses the NUMBER of readings per vital, so sending fewer readings than were
charted changes the score. Requests with far fewer readings than typical are flagged
`low_count`.
"""
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from src.config import load_config

CFG = load_config()
AGE_MIN, AGE_MAX = CFG["valid_ranges"]["Age"]
TIME_RE = re.compile(r"^(\d{1,3}):([0-5]\d)$")

QualityFlag = Literal["ok", "low_count", "missing_vitals", "insufficient_data"]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    time: str = Field(description="HH:MM since ICU admission, e.g. '07:30' or '24:00'")
    parameter: Literal["HR", "RespRate", "Temp"]
    value: float = Field(allow_inf_nan=False, description="HR beats/min, RespRate breaths/min, Temp °C")

    @field_validator("time")
    @classmethod
    def check_time(cls, v: str) -> str:
        if not TIME_RE.match(v):
            raise ValueError("time must be HH:MM with minutes 00-59, e.g. '07:30'")
        return v


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    record_id: str | None = Field(default=None, max_length=64,
                                  description="Echoed back, never logged")
    age: float = Field(ge=AGE_MIN, le=AGE_MAX, description="Years. Training data is adults only.")
    # StrictInt, not Literal[0, 1]: Literal accepts true/false because bool is an int in Python.
    gender: Annotated[StrictInt, Field(ge=0, le=1)] | None = Field(
        default=None, description="0 female, 1 male, null unknown")
    icu_type: Annotated[StrictInt, Field(ge=1, le=4)] = Field(
        description="1 coronary, 2 cardiac surgery recovery, 3 medical, 4 surgical")
    observations: list[Observation] = Field(
        default_factory=list, max_length=5000,
        description="Every charted HR, RespRate and Temp reading from the first 24 h. "
                    "Do not pre-aggregate: the number of readings is a model input.")


class PredictResponse(BaseModel):
    record_id: str | None
    probability: float
    high_risk: bool | None = Field(
        description="null when data_quality_flag is insufficient_data: no alert is raised "
                    "from age, sex and ICU type alone")
    threshold: float
    model_version: str
    data_version: str
    data_quality_flag: QualityFlag
    missing_vitals: list[str]
    low_count_vitals: list[str]
    warnings: list[str]
    request_id: str
